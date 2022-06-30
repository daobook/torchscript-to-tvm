import tvm
from tvm import relay, auto_scheduler
from tvm.runtime.vm import VirtualMachine

import numpy as np
import cv2

import torch
from torch import nn


def do_trace(model, inp):
    model_trace = torch.jit.trace(model, inp)
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


from yolort.models import yolov5l


model_func = yolov5l(export_friendly=True, pretrained=True)


model = TraceWrapper(model_func)

model.eval()
in_size = 512
inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(8, 3, in_size, in_size)))

with torch.no_grad():
    out = model(inp)
    script_module = do_trace(model, inp)

img = np.random.randn(8, 3, 512, 512).astype("float32")

input_name = "input0"
shape_list = [(input_name, inp.shape)]
mod, params = relay.frontend.from_pytorch(script_module, shape_list)

with tvm.transform.PassContext(opt_level=3):
    desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
    seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
    mod = seq(mod)

target = "cuda"
log_file = "yolov5l.log"

def auto_schedule():
    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)
    return


def run():
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            vm_exec = relay.vm.compile(mod, target=target, params=params)

    ctx = tvm.device(target, 0)
    vm = VirtualMachine(vm_exec, ctx)
    vm.set_input("main", **{input_name: img})
    tvm_res = vm.run()

    with torch.no_grad():
        torch_res = model(torch.from_numpy(img))

    ftimer = vm.module.time_evaluator("invoke", ctx, number=1, repeat=50)
    print(ftimer("main"))

    # for i in range(3):
    #     print(np.max(np.abs(torch_res[i].numpy() - tvm_res[i].asnumpy())))

auto_schedule()
# run()
