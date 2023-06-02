# %%
from models.vanillanet import *
import time
import torch
import onnxruntime as ort
import numpy as np
import onnx

# %%
import itertools
import subprocess
from argparse import Namespace
from dataclasses import dataclass
from typing import List
import pandas as pd
import re


@dataclass
class BenchmarkResult:
    stdout: str = ''
    stderr: str = ''
    avg_latency: float = 0.
    throughput: float = 0.


def run_benchmark(cmd):
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        p.wait()
        stdout = p.stdout.read().decode()
        stderr = p.stderr.read().decode()
        # print(stdout)
        # print(stderr)
        stdout = stdout.strip()
        avg_line = filter(None, stdout.split('\n')[-4].split())
        throughput_line = filter(None, stdout.split('\n')[-1].split())
        avg_line = list(avg_line)
        throughput_line = list(throughput_line)
        assert 'Average:' in avg_line and 'Throughput:' in throughput_line

        return BenchmarkResult(
            stdout=stdout,
            stderr=stderr,
            avg_latency=float(list(avg_line)[-2]),
            throughput=float(list(throughput_line)[-2]),
        )

# %%


def get_vanillanet_model(model_cls) -> torch.nn.Module:
    net = model_cls().cuda()
    net.eval()
    net.switch_to_deploy()
    return net

# %%


def job(net, name_str):
    result = {}
    img = torch.rand((1, 3, 224, 224))

    # latency
    net = net.eval()
    img = img.cuda()
    net = net.cuda()
    with torch.no_grad():
        for i in range(50):
            net(img)
        torch.cuda.synchronize()
        t = time.time()
        for i in range(1000):
            net(img)
            torch.cuda.synchronize()
    result['latency_cuda_torch'] = (time.time() - t)

    # torch output
    with torch.no_grad():
        torch_output = net(img)
        if not isinstance(torch_output, torch.Tensor):
            torch_output = torch_output.logits
        torch_output = torch_output.cpu().view(-1)

    onnx_path = f"/tmp/yujiepan/{name_str}.onnx"
    model = net.eval().cpu()
    torch.onnx.export(model,
                      img.cpu(),
                      onnx_path,
                      verbose=False,
                      opset_version=13,
                      do_constant_folding=True,
                      )

    onnx_model = onnx.load_model(onnx_path)
    sess = ort.InferenceSession(onnx_model.SerializeToString(), providers=['CPUExecutionProvider'])
    # sess.set_providers(['CPUExecutionProvider'])
    # sess.set_providers(['CUDAExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = sess.run([output_name], {input_name : img.cpu().numpy()})
    onnx_output = torch.tensor(output).view(-1)
    torch.testing.assert_close(onnx_output, torch_output, atol=1e-4, rtol=1e-4)

    result['latency'] = run_benchmark(cmd=f'benchmark_app -m {onnx_path} -niter 2000 -hint latency').avg_latency
    result['throughput'] = run_benchmark(cmd=f'benchmark_app -m {onnx_path} -hint throughput -t 35').throughput
    return result


# %%
from transformers import AutoFeatureExtractor, SwinForImageClassification
import torchvision
import repvgg
import pandas as pd

all_results = []


def get_list():
    yield (torchvision.models.mobilenet_v3_large(), 'mobilenet_v3_large')
    yield (SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224'), 'swin_t')
    yield (SwinForImageClassification.from_pretrained('microsoft/swin-small-patch4-window7-224'), 'swin_s')
    yield (repvgg.create_RepVGG_B3(deploy=True), 'create_RepVGG_B3')
    yield (torchvision.models.resnet50(), 'resnet50')
    yield (get_vanillanet_model(vanillanet_9), 'vanillanet_9')
    yield (get_vanillanet_model(vanillanet_12), 'vanillanet_12')
    # yield    (get_model(vanillanet_13_x1_5_ada_pool), 'vanillanet_13_x1_5_ada_pool')
    yield (get_vanillanet_model(vanillanet_13_x1_5), 'vanillanet_13_x1_5')


for model, name_str in get_list():
    result = job(model, name_str)
    result['name'] = name_str
    all_results.append(result)
    pd.DataFrame(all_results).to_csv('result.csv')
