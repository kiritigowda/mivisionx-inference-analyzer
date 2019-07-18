[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# MIVisionX Python Inference Analyzer

[MIVisionX](https://gpuopen-professionalcompute-libraries.github.io/MIVisionX/) Inference Application using pre-trained `ONNX`/`NNEF`/`Caffe` models to analyze images

<p align="center"><img width="60%" src="data/images/inference_analyzer.gif" /></p>

Pre-trained models in [ONNX](https://onnx.ai/), [NNEF](https://www.khronos.org/nnef), & [Caffe](http://caffe.berkeleyvision.org/) formats are supported by MIVisionX. The app first converts the pre-trained models to AMD Neural Net Intermediate Representation (NNIR), once the model has been translated into AMD NNIR (AMD's internal open format), the Optimizer goes through the NNIR and applies various optimizations which would allow the model to be deployed on to target hardware most efficiently. Finally, AMD NNIR is converted into OpenVX C code, which is compiled and wrapped with a python API to run on any targeted hardware.

## Prerequisites

* Ubuntu `16.04`/`18.04` or CentOS `7.5`/`7.6`
* [ROCm supported hardware](https://rocm.github.io/ROCmInstall.html#hardware-support) 
	* AMD Radeon GPU or AMD APU required
* Latest [ROCm](https://github.com/RadeonOpenCompute/ROCm#installing-from-amd-rocm-repositories)
* Build & Install [MIVisionX](https://github.com/GPUOpen-ProfessionalCompute-Libraries/MIVisionX#linux-1)

## Usage
````
usage: mivisionx_inference_analyzer.py [-h] 
                             	       --model_format MODEL_FORMAT 
                                       --model_name MODEL_NAME 
                                       --model MODEL 
                                       --model_input_dims MODEL_INPUT_DIMS 
                                       --model_output_dims MODEL_OUTPUT_DIMS 
                                       --label LABEL 
                                       --output_dir OUTPUT_DIR 
                                       --image_dir IMAGE_DIR
                                       [--image_val IMAGE_VAL] 
                                       [--hierarchy HIERARCHY]
                                       [--add ADD] 
                                       [--multiply MULTIPLY]
                                       [--replace REPLACE] 
                                       [--verbose VERBOSE]

````
### Usage Help

```
  -h, --help            show this help message and exit
  --model_format        pre-trained model format, options:caffe/onnx/nnef [required]
  --model_name          model name                                        [required]
  --model               pre_trained model file                            [required]
  --model_input_dims    c,h,w - channel,height,width                      [required]
  --model_output_dims   c,h,w - channel,height,width                      [required]
  --label               labels text file                                  [required]
  --output_dir          output dir to store ADAT results                  [required]
  --image_dir           image directory for analysis                      [required]
  --image_val           image list with ground truth                      [optional]
  --hierarchy           AMD proprietary hierarchical file                 [optional]
  --add                 input preprocessing factor            [optional - default:0]
  --multiply            input preprocessing factor            [optional - default:1]
  --replace             replace/overwrite model              [optional - default:no]
  --verbose             verbose                              [optional - default:no]

```
## Samples

### Sample 1 - Using Pre-Trained ONNX Model

#### Run SqueezeNet on sample images

* **Step 1:** Clone MIVisionX Inference Analyzer Project

	````
	% cd && mkdir sample-1 && cd sample-1
	% https://github.com/kiritigowda/MIVisionX-inference-analyzer.git
	````

	**Note:**
	* MIVisionX needs to be pre-installed
	* MIVisionX Model Compiler & Optimizer scripts are at `/opt/rocm/mivisionx/model_compiler/python/`
	* ONNX model conversion requires ONNX install using `pip install onnx`	

* **Step 2:** Download pre-trained SqueezeNet ONNX model from [ONNX Model Zoo](https://github.com/onnx/models#open-neural-network-exchange-onnx-model-zoo) - [SqueezeNet Model](https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz)
	````
	% wget https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz
	% tar -xvf squeezenet.tar.gz
	````
	**Note:** pre-trained model - `squeezenet/model.onnx` 

* **Step 3:** Use the command below to run the inference analyzer

	* View inference analyzer usage
	```
	% cd ~/sample-1/MIVisionX-inference-analyzer/
	% python mivisionx_inference_analyzer.py -h
	```
	
	* Run SqueezeNet Inference Analyzer
	```
	% python mivisionx_inference_analyzer.py --model_format onnx --model_name SqueezeNet --model ~/sample-1/squeezenet/model.onnx --model_input_dims 3,224,224 --model_output_dims 1000,1,1 --label ./sample/labels.txt --output_dir ~/sample-1/ --image_dir ./sample/AMD-tinyDataSet --image_val ./sample/AMD-tinyDataSet-val.txt --hierarchy ./sample/hierarchy.csv --replace yes
	```
