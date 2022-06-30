import numpy as np
import torch

import tvm
from tvm import relay
from tvm.relay.frontend.pytorch import from_pytorch
from tvm.relay.prelude import Prelude
from tvm.runtime.container import ADT, tuple_object


def vmobj_to_list(o, dtype="float32"):
    if isinstance(o, tvm.nd.NDArray):
        return [o]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f, dtype))
        return result
    else:
        raise RuntimeError(f"Unknown object type: {type(o)}")


def assert_equal(tvm_result, torch_result):
    if isinstance(torch_result, (tuple, list)):
        assert isinstance(tvm_result, list)
        for tvm_res, pt_res in zip(tvm_result, torch_result):
            assert_equal(tvm_res, pt_res)
    elif isinstance(torch_result, torch.Tensor):
        print(np.max(np.abs(tvm_result.asnumpy() - torch_result.numpy())))
        tvm.testing.assert_allclose(tvm_result.asnumpy(), torch_result.numpy(),
                                    rtol=1e-5, atol=1e-5)
    else:
        tvm_res = np.asscalar(tvm_result.asnumpy())
        print(abs(torch_result - tvm_res))
        assert torch_result == tvm_res


def run_and_compare(mod, params, pt_result):
    executor = relay.create_executor("vm", mod=mod, ctx=tvm.cpu(0), target="llvm")
    # executable = executor.executable
    # code, lib = executable.save()
    # print(lib.get_source("s"))
    evaluator = executor.evaluate()

    exec_res = evaluator(**params)

    def flatten(nested):
        res = []
        for r in nested:
            if isinstance(r, torch.Tensor):
                res.append(r)
            else:
                res.extend(flatten(r))
        return res

    if isinstance(exec_res, tvm.runtime.container.ADT):
        assert not isinstance(pt_result, torch.Tensor)
        tvm_res = vmobj_to_list(exec_res)
        torch_res = flatten(pt_result)
    else:
        tvm_res = exec_res
        torch_res = pt_result

    assert_equal(tvm_res, torch_res)


def simple_rnn_test():
    class DecisionGate(torch.nn.Module):
        def forward(self, x):
            return x if x.sum() > 0 else -x

    class Cell(torch.nn.Module):
        def __init__(self, dg):
            super(Cell, self).__init__()
            self.dg = dg
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x, h):
            new_h = torch.tanh(self.dg(self.linear(x)) + h)
            return new_h, new_h

    class RNNLoop(torch.nn.Module):
        def __init__(self):
            super().__init__()
            x = torch.rand(10, 4, dtype=torch.float)
            h = torch.rand(10, 4, dtype=torch.float)
            self.cell = torch.jit.trace(Cell(DecisionGate()), (x, h))

        def forward(self, xs):
            h = torch.zeros(10, 4, dtype=torch.float)
            y = torch.zeros(10, 4, dtype=torch.float)
            for i in range(xs.size(0)):
                y, h = self.cell(xs[i], h)
            return y

    raw_model = RNNLoop()
    script_module = torch.jit.script(raw_model)
    input_name = "input"
    input_shapes = [(input_name, (10, 10, 4))]

    mod, params = from_pytorch(script_module, input_shapes, {})

    inp = torch.randn(input_shapes[0][1], dtype=torch.float)
    with torch.no_grad():
        pt_result = raw_model(inp.clone())

    params[input_name] = inp.numpy()

    run_and_compare(mod, params, pt_result)


def convert_list_to_vmobj(py_lst):
    def wrap_nd_array(arr):
        return tvm.nd.array(arr, ctx=tvm.cpu(0))

    mod = tvm.IRModule()
    prelude = Prelude(mod)
    adt_lst = ADT(prelude.nil.tag, [])
    for elem in reversed(py_lst):
        if isinstance(elem, np.ndarray):
            vmobj = wrap_nd_array(elem)
        elif isinstance(elem, tuple):
            vmobj = tuple_object([wrap_nd_array(e) for e in elem])
        elif isinstance(elem, list):
            vmobj = convert_list_to_vmobj(elem)
        adt_lst = ADT(prelude.cons.tag, [vmobj, adt_lst])
    return adt_lst


def custom_lstm_test():
    input_name = "input"
    states_name = "states"
    seq_len = 5
    batch = 2
    input_size = 3
    hidden_size = 4
    num_layers = 3
    state_tensor_shape = (batch, hidden_size)

    inp = torch.randn(seq_len, batch, input_size)

    input_shapes = [(input_name, (seq_len, batch, input_size)),
                    (states_name, (state_tensor_shape, state_tensor_shape))]

    input_shapes_stacked = [(input_name, (seq_len, batch, input_size)),
                            (states_name, [(state_tensor_shape, state_tensor_shape),
                                           (state_tensor_shape, state_tensor_shape)])]

    input_shapes_stacked_bidir = [(input_name, (seq_len, batch, input_size)),
                                  (states_name, [[(state_tensor_shape,
                                                   state_tensor_shape)
                                                  for _ in range(2)]
                                                 for _ in range(num_layers)])]

    states = [(torch.randn(state_tensor_shape),
               torch.randn(state_tensor_shape))
              for _ in range(num_layers)]

    bidir_states = [(torch.randn(state_tensor_shape),
                     torch.randn(state_tensor_shape))
                    for _ in range(2)]

    stacked_bidir_states = [[(torch.randn(state_tensor_shape),
                              torch.randn(state_tensor_shape))
                             for _ in range(2)]
                            for _ in range(num_layers)]

    from custom_lstms import lstm, stacked_lstm, bidir_lstm, stacked_bidir_lstm

    models = [
      (lstm(input_size, hidden_size).eval(), states[0], input_shapes),
      (stacked_lstm(input_size, hidden_size, num_layers).eval(), states, input_shapes_stacked),
      (bidir_lstm(input_size, hidden_size).eval(), bidir_states, input_shapes_stacked),
      (stacked_bidir_lstm(input_size, hidden_size, num_layers).eval(),
       stacked_bidir_states, input_shapes_stacked_bidir)
    ]

    for (raw_model, states, input_shapes) in models:
        script_module = torch.jit.script(raw_model)
        mod, params = from_pytorch(script_module, input_shapes)
        # print(mod["main"])

        with torch.no_grad():
            pt_result = raw_model(inp.clone(), states)

        params[input_name] = inp.numpy()

        if isinstance(states, tuple):
            states_np = tuple(st.numpy() for st in states)
        elif isinstance(states, list) and isinstance(states[0], torch.Tensor):
            states_np = [st.numpy() for st in states]
        elif isinstance(states, list) and isinstance(states[0], tuple):
            states_np = [tuple(st.numpy() for st in states[i])
                         for i in range(len(states))]
        elif isinstance(states, list) and isinstance(states[0], list):
            states_np = [[tuple(st.numpy() for st in states)
                         for states in states[layer]]
                         for layer in range(num_layers)]
        else:
            assert False

        if isinstance(states_np, list):
            params[states_name] = convert_list_to_vmobj(states_np)
        else:
            params[states_name] = states_np

        run_and_compare(mod, params, pt_result)


custom_lstm_test()
simple_rnn_test()
