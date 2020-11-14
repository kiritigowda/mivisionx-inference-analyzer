from inference_control import *
from numpy.ctypeslib import ndpointer
import argparse
import os
import sys
import cv2
import numpy as np
import numpy
import time
import ctypes
import pandas
from collections import Counter
from PIL import Image

__author__ = "Kiriti Nagesh Gowda"
__copyright__ = "Copyright 2019, AMD MIVisionX"
__credits__ = ["Mike Schmit; Hansel Yang; Lakshmi Kumar;"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Kiriti Nagesh Gowda"
__email__ = "Kiriti.NageshGowda@amd.com"
__status__ = "Shipping"
__script_name__ = "MIVisionX Inference Analyzer"

# global variables
FP16inference = False
verbosePrint = False
labelNames = None
colors = [
    (0, 153, 0),        # Top1
    (153, 153, 0),      # Top2
    (153, 76, 0),       # Top3
    (0, 128, 255),      # Top4
    (255, 102, 102),    # Top5
]

# AMD Neural Net python wrapper
class AnnAPI:
    def __init__(self, library):
        self.lib = ctypes.cdll.LoadLibrary(library)
        self.annQueryInference = self.lib.annQueryInference
        self.annQueryInference.restype = ctypes.c_char_p
        self.annQueryInference.argtypes = []
        self.annCreateInference = self.lib.annCreateInference
        self.annCreateInference.restype = ctypes.c_void_p
        self.annCreateInference.argtypes = [ctypes.c_char_p]
        self.annReleaseInference = self.lib.annReleaseInference
        self.annReleaseInference.restype = ctypes.c_int
        self.annReleaseInference.argtypes = [ctypes.c_void_p]
        self.annCopyToInferenceInput = self.lib.annCopyToInferenceInput
        self.annCopyToInferenceInput.restype = ctypes.c_int
        self.annCopyToInferenceInput.argtypes = [ctypes.c_void_p, ndpointer(
            ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_bool]
        self.annCopyFromInferenceOutput = self.lib.annCopyFromInferenceOutput
        self.annCopyFromInferenceOutput.restype = ctypes.c_int
        self.annCopyFromInferenceOutput.argtypes = [ctypes.c_void_p, ndpointer(
            ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
        self.annRunInference = self.lib.annRunInference
        self.annRunInference.restype = ctypes.c_int
        self.annRunInference.argtypes = [ctypes.c_void_p, ctypes.c_int]
        print('OK: AnnAPI found "' + self.annQueryInference().decode("utf-8") +
              '" as configuration in ' + library)

# classifier definition
class annieObjectWrapper():
    def __init__(self, annpythonlib, weightsfile):
        self.api = AnnAPI(annpythonlib)
        input_info, output_info, empty = self.api.annQueryInference().decode("utf-8").split(';')
        input, name, n_i, c_i, h_i, w_i = input_info.split(',')
        outputCount = output_info.split(",")
        stringcount = len(outputCount)
        if stringcount == 6:
            output, opName, n_o, c_o, h_o, w_o = output_info.split(',')
        else:
            output, opName, n_o, c_o = output_info.split(',')
            h_o = '1'
            w_o = '1'
        self.hdl = self.api.annCreateInference(weightsfile.encode('utf-8'))
        self.dim = (int(w_i), int(h_i))
        self.outputDim = (int(n_o), int(c_o), int(h_o), int(w_o))

    def __del__(self):
        self.api.annReleaseInference(self.hdl)

    def runInference(self, img, out):
        # create input.f32 file
        img_r = img[:, :, 0]
        img_g = img[:, :, 1]
        img_b = img[:, :, 2]
        img_t = np.concatenate((img_r, img_g, img_b), 0)
        # copy input f32 to inference input
        status = self.api.annCopyToInferenceInput(self.hdl, np.ascontiguousarray(
            img_t, dtype=np.float32), (img.shape[0]*img.shape[1]*3*4), 0)
        if(status):
            print('ERROR: annCopyToInferenceInput Failed ')
        # run inference
        status = self.api.annRunInference(self.hdl, 1)
        if(status):
            print('ERROR: annRunInference Failed ')
        # copy output f32
        status = self.api.annCopyFromInferenceOutput(
            self.hdl, np.ascontiguousarray(out, dtype=np.float32), out.nbytes)
        if(status):
            print('ERROR: annCopyFromInferenceOutput Failed ')
        return out

    def classify(self, img):
        # create output.f32 buffer
        out_buf = bytearray(
            self.outputDim[0]*self.outputDim[1]*self.outputDim[2]*self.outputDim[3]*4)
        out = np.frombuffer(out_buf, dtype=numpy.float32)
        # run inference & receive output
        output = self.runInference(img, out)
        return output

# process classification output function
def processClassificationOutput(inputImage, modelName, modelOutput):
    # post process output file
    start = time.time()
    softmaxOutput = np.float32(modelOutput)
    topIndex = []
    topLabels = []
    topProb = []
    for x in softmaxOutput.argsort()[-5:]:
        topIndex.append(x)
        topLabels.append(labelNames[x])
        topProb.append(softmaxOutput[x])
    end = time.time()
    if(verbosePrint):
        print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms'

    # display output
    start = time.time()
    # initialize the result image
    resultImage = np.zeros((250, 525, 3), dtype="uint8")
    resultImage.fill(255)
    cv2.putText(resultImage, 'MIVisionX Object Classification',
                (25,  25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    topK = 1
    for i in reversed(range(5)):
        txt = topLabels[i].decode('utf-8')[:-1]
        conf = topProb[i]
        txt = 'Top'+str(topK)+':'+txt+' '+str(int(round((conf*100), 0)))+'%'
        size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        t_height = size[0][1]
        textColor = (colors[topK - 1])
        cv2.putText(resultImage, txt, (45, t_height+(topK*30+40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)
        topK = topK + 1
    end = time.time()
    if(verbosePrint):
        print '%30s' % 'Processed results image in ', str((end - start)*1000), 'ms'

    return resultImage, topIndex, topProb


# MIVisionX Classifier
if __name__ == '__main__':

    if len(sys.argv) == 1:
        app = QtGui.QApplication(sys.argv)
        panel = inference_control()
        app.exec_()
        modelFormat = (str)(panel.model_format)
        modelName = (str)(panel.model_name)
        modelLocation = (str)(panel.model)
        modelInputDims = (str)(panel.input_dims)
        modelOutputDims = (str)(panel.output_dims)
        label = (str)(panel.label)
        outputDir = (str)(panel.output)
        imageDir = (str)(panel.image)
        imageVal = (str)(panel.val)
        hierarchy = (str)(panel.hier)
        inputAdd = (str)(panel.add)
        inputMultiply = (str)(panel.multiply)
        fp16 = (str)(panel.fp16)
        replaceModel = (str)(panel.replace)
        verbose = (str)(panel.verbose)
        resize_inter = (int)(panel.resize_inter)
        display_option = (int)(panel.display_option)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_format',		type=str, required=True,
                            help='pre-trained model format, options:caffe/onnx/nnef [required]')
        parser.add_argument('--model_name',			type=str, required=True,
                            help='model name                             [required]')
        parser.add_argument('--model',				type=str, required=True,
                            help='pre_trained model file/folder          [required]')
        parser.add_argument('--model_input_dims',	type=str, required=True,
                            help='c,h,w - channel,height,width           [required]')
        parser.add_argument('--model_output_dims',	type=str, required=True,
                            help='c,h,w - channel,height,width           [required]')
        parser.add_argument('--label',				type=str, required=True,
                            help='labels text file                       [required]')
        parser.add_argument('--output_dir',			type=str, required=True,
                            help='output dir to store ADAT results       [required]')
        parser.add_argument('--image_dir',			type=str, required=True,
                            help='image directory for analysis           [required]')
        parser.add_argument('--image_val',			type=str, default='',
                            help='image list with ground truth           [optional]')
        parser.add_argument('--hierarchy',			type=str, default='',
                            help='AMD proprietary hierarchical file      [optional]')
        parser.add_argument('--add',				type=str, default='',
                            help='input preprocessing factor [optional - default:[0,0,0]]')
        parser.add_argument('--multiply',			type=str, default='',
                            help='input preprocessing factor [optional - default:[1,1,1]]')
        parser.add_argument('--fp16',				type=str, default='no',
                            help='quantize to FP16 			 [optional - default:no]')
        parser.add_argument('--resize_option',		type=int, default=0,
                            help='image resize interpolation [optional - default:0 range[0 - 5]]')
        parser.add_argument('--replace',			type=str, default='no',
                            help='replace/overwrite model    [optional - default:no]')
        parser.add_argument('--verbose',			type=str, default='no',
                            help='verbose                    [optional - default:no]')
        parser.add_argument('--display_option',		type=int, default=1,
                            help='image resize interpolation [optional - default:1 range[0 - 2]]')
        args = parser.parse_args()

        # get arguments
        modelFormat = args.model_format
        modelName = args.model_name
        modelLocation = args.model
        modelInputDims = args.model_input_dims
        modelOutputDims = args.model_output_dims
        label = args.label
        outputDir = args.output_dir
        imageDir = args.image_dir
        imageVal = args.image_val
        hierarchy = args.hierarchy
        inputAdd = args.add
        inputMultiply = args.multiply
        fp16 = args.fp16
        replaceModel = args.replace
        verbose = args.verbose
        resize_inter = args.resize_option
        display_option = args.display_option

    # set verbose print
    if(verbose != 'no'):
        verbosePrint = True

    # set fp16 inference turned on/off
    if(fp16 != 'no'):
        FP16inference = True

    # Set Display Option
    if display_option not in (0, 1, 2):
        print("WARNING: Display [options: 0:OFF, 1:MIN, or 2:ALL]")
        display_option = 2

    # set paths
    modelCompilerPath = '/opt/rocm/mivisionx/model_compiler/python'
    ADATPath = '/opt/rocm/mivisionx/toolkit/amd_data_analysis_toolkit/classification'
    setupDir = '~/.mivisionx-inference-analyzer'
    analyzerDir = os.path.expanduser(setupDir)
    modelDir = analyzerDir+'/'+modelName+'_dir'
    nnirDir = modelDir+'/nnir-files'
    openvxDir = modelDir+'/openvx-files'
    modelBuildDir = modelDir+'/build'
    adatOutputDir = os.path.expanduser(outputDir)
    inputImageDir = os.path.expanduser(imageDir)
    trainedModel = os.path.expanduser(modelLocation)
    labelText = os.path.expanduser(label)
    hierarchyText = os.path.expanduser(hierarchy)
    imageValText = os.path.expanduser(imageVal)
    pythonLib = modelBuildDir+'/libannpython.so'
    weightsFile = openvxDir+'/weights.bin'
    finalImageResultsFile = modelDir+'/imageResultsFile.csv'
    imageSizeCountFile = modelDir+'/originalImageSizeCounter.csv'
    imageSizeFile = modelDir+'/originalImageSizes.csv'
    imageSizeCountGraph = modelDir+'/originalImageSizePlot.png'

    # get input & output dims
    str_c_i, str_h_i, str_w_i = modelInputDims.split(',')
    c_i = int(str_c_i)
    h_i = int(str_h_i)
    w_i = int(str_w_i)
    str_c_o, str_h_o, str_w_o = modelOutputDims.split(',')
    c_o = int(str_c_o)
    h_o = int(str_h_o)
    w_o = int(str_w_o)

    # cv resize interpolation
    interpolation_methond = cv2.INTER_LINEAR
    if(resize_inter == 1):
        # nearest neighbor interpolation
        interpolation_method = cv2.INTER_NEAREST
    elif(resize_inter == 2 or resize_inter == 0):
        # bilinear interpolation
        interpolation_method = cv2.INTER_LINEAR
    elif(resize_inter == 3):
        # bicubic interpolation
        interpolation_method = cv2.INTER_CUBIC
    elif(resize_inter == 4):
        # resampling using pixel area relation.
        # It may be a preferred method for image decimation, as it gives moire'-free results.
        # But when the image is zoomed, it is similar to the INTER_NEAREST method.
        interpolation_method = cv2.INTER_AREA
    elif(resize_inter == 5):
        # Lanczos interpolation over 8x8 neighborhood
        interpolation_method = cv2.INTER_LANCZOS4
    else:
        print("\nResize interpolation only supports 5 methods - default:INTER_LINEAR\n")

    # input pre-processing values
    Ax = [0, 0, 0]
    if(inputAdd != ''):
        Ax = [float(item) for item in inputAdd.strip("[]").split(',')]
    Mx = [1, 1, 1]
    if(inputMultiply != ''):
        Mx = [float(item) for item in inputMultiply.strip("[]").split(',')]

    # check pre-trained model
    if(not os.path.isfile(trainedModel) and modelFormat != 'nnef'):
        print("\nPre-Trained Model not found, check argument --model\n")
        quit()

    # check for label file
    if (not os.path.isfile(labelText)):
        print("\nlabels.txt not found, check argument --label\n")
        quit()
    else:
        fp = open(labelText, 'r')
        labelNames = fp.readlines()
        fp.close()

    # MIVisionX setup
    if(os.path.exists(analyzerDir)):
        print("\nMIVisionX Inference Analyzer\n")
        # replace old model or throw error
        if(replaceModel == 'yes'):
            os.system('rm -rf '+modelDir)
        elif(os.path.exists(modelDir)):
            print("OK: Model exists")

    else:
        print("\nMIVisionX Inference Analyzer Created\n")
        os.system('(cd ; mkdir .mivisionx-inference-analyzer)')

    # Setup Text File for Demo
    if (not os.path.isfile(analyzerDir + "/setupFile.txt")):
        f = open(analyzerDir + "/setupFile.txt", "w")
        f.write(modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' +
                imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ", "") + ';' + str(Mx).strip('[]').replace(" ", "") + ';' + fp16 + ';' + replaceModel + ';' + verbose)
        f.close()
    else:
        count = len(open(analyzerDir + "/setupFile.txt").readlines())
        if count < 10:
            with open(analyzerDir + "/setupFile.txt", "r") as fin:
                data = fin.read().splitlines(True)
                modelList = []
                for i in range(len(data)):
                    modelList.append(data[i].split(';')[1])
                if modelName not in modelList:
                    f = open(analyzerDir + "/setupFile.txt", "a")
                    f.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' +
                            imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ", "") + ';' + str(Mx).strip('[]').replace(" ", "") + ';' + fp16 + ';' + replaceModel + ';' + verbose)
                    f.close()
        else:
            with open(analyzerDir + "/setupFile.txt", "r") as fin:
                data = fin.read().splitlines(True)
            delModelName = data[0].split(';')[1]
            delmodelPath = analyzerDir + '/' + delModelName + '_dir'
            if(os.path.exists(delmodelPath)):
                os.system('rm -rf ' + delmodelPath)
            with open(analyzerDir + "/setupFile.txt", "w") as fout:
                fout.writelines(data[1:])
            with open(analyzerDir + "/setupFile.txt", "a") as fappend:
                fappend.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir +
                              ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ", "") + ';' + str(Mx).strip('[]').replace(" ", "") + ';' + fp16 + ';' + replaceModel + ';' + verbose)
                fappend.close()

    # Compile Model and generate python .so files
    if (replaceModel == 'yes' or not os.path.exists(modelDir)):
        os.system('mkdir '+modelDir)
        if(os.path.exists(modelDir)):
            # convert to NNIR
            if(modelFormat == 'caffe'):
                os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/caffe_to_nnir.py ' +
                          trainedModel+' nnir-files --input-dims 1,'+modelInputDims+' )')
            elif(modelFormat == 'onnx'):
                os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/onnx_to_nnir.py ' +
                          trainedModel+' nnir-files --input-dims 1,'+modelInputDims+' )')
            elif(modelFormat == 'nnef'):
                os.system('(cd '+modelDir+'; python '+modelCompilerPath +
                          '/nnef_to_nnir.py '+trainedModel+' nnir-files )')
            else:
                print(
                    "ERROR: Neural Network Format Not supported, use caffe/onnx/nnef in arugment --model_format")
                quit()
            # convert the model to FP16
            if(FP16inference):
                os.system('(cd '+modelDir+'; python '+modelCompilerPath +
                          '/nnir_update.py --convert-fp16 1 --fuse-ops 1 nnir-files nnir-files)')
                print("\nModel Quantized to FP16\n")
            # convert to openvx
            if(os.path.exists(nnirDir)):
                os.system('(cd '+modelDir+'; python '+modelCompilerPath +
                          '/nnir_to_openvx.py nnir-files openvx-files)')
            else:
                print("ERROR: Converting Pre-Trained model to NNIR Failed")
                quit()

            # build model
            if(os.path.exists(openvxDir)):
                os.system('mkdir '+modelBuildDir)
            else:
                print("ERROR: Converting NNIR to OpenVX Failed")
                quit()
    # build model
    os.system('(cd '+modelBuildDir +
              '; cmake ../openvx-files; make; ./anntest ../openvx-files/weights.bin )')
    # verify
    annTestExe = os.path.expanduser(modelBuildDir+'/anntest')
    if (not os.path.isfile(annTestExe)):
        print(
            "\nERROR: Failed to Create Neural Net Executable, check MIVisionX Installation")
        quit()
    else:
        print("\nSUCCESS: Converting Pre-Trained model to MIVisionX Runtime successful\n")

    # opencv display window
    if(display_option == 2):
        windowInput = "MIVisionX Inference Analyzer - Input Image"
        windowResult = "MIVisionX Inference Analyzer - Results"
        cv2.namedWindow(windowInput, cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(windowInput, 800, 800)
    elif(display_option >= 1):
        windowProgress = "MIVisionX Inference Analyzer - Progress"
    else:
        print("\nINFO: Display Option set to 0 - Display OFF\n")

    # create inference classifier
    classifier = annieObjectWrapper(pythonLib, weightsFile)

    # check for image val text
    totalImages = 0
    if(imageVal == ''):
        print(
            "\nFlow without Image Validation Text..Creating a file with no ground truths\n")
        imageList = os.listdir(inputImageDir)
        imageList.sort()
        imageValText = os.getcwd() + '/imageValTxt.txt'
        fp = open(imageValText, 'w')
        for imageFile in imageList:
            fp.write(imageFile + " -1" + "\n")

    if (not os.path.isfile(imageValText)):
        print("\nImage Validation Text not found, check argument --image_val\n")
        quit()
    else:
        fp = open(imageValText, 'r')
        imageValidation = fp.readlines()
        fp.close()
        totalImages = len(imageValidation)

    # original std out location
    orig_stdout = sys.stdout
    # setup results output file
    sys.stdout = open(finalImageResultsFile, 'w')
    print('Image File Name,Ground Truth Label,Output Label 1,Output Label 2,Output Label 3,\
		Output Label 4,Output Label 5,Prob 1,Prob 2,Prob 3,Prob 4,Prob 5,Original Image Scale')
    sys.stdout = orig_stdout

    # process images
    correctTop5 = 0
    correctTop1 = 0
    wrong = 0
    noGroundTruth = 0
    originalImageSizes = ["" for x in range(totalImages)]
    for x in range(totalImages):
        imageFileName, grountTruth = imageValidation[x].decode(
            "utf-8").split(' ')
        groundTruthIndex = int(grountTruth)
        imageFile = os.path.expanduser(inputImageDir+'/'+imageFileName)
        if (not os.path.isfile(imageFile)):
            print 'Image File - '+imageFile+' not found'
            quit()
        else:
            # read image
            start = time.time()
            # tmp PIL fix
            pil_image = Image.open(imageFile).convert('RGB')
            opencv_image = numpy.array(pil_image)
            opencv_image = opencv_image[:, :, ::-1].copy()
            # handle back to OpenCV
            #frame = cv2.imread(imageFile,0)
            frame = opencv_image
            assert not isinstance(frame, type(
                None)), 'ERROR: Image Not found:'+imageFile
            end = time.time()
            if(verbosePrint):
                print '%30s' % 'Read Image in ', str((end - start)*1000), 'ms'

            # resize image
            start = time.time()
            original_width = frame.shape[1]
            original_height = frame.shape[0]
            originalImageSizes[x] = format(
                original_width, '05d')+'x'+format(original_height, '05d')
            ImageScaleFactor = float(
                original_width * original_height)/(w_i * h_i)
            resizedFrame = cv2.resize(
                frame, (w_i, h_i), interpolation=interpolation_method)
            end = time.time()
            if(verbosePrint):
                print '%30s' % 'Original WxH:'+str(original_width)+'x'+str(original_height)+' Resized WxH:'+str(w_i)+'x'+str(h_i)
                print '%30s' % 'Input Image Resized in ', str((end - start)*1000), 'ms'

            # pre-process input
            start = time.time()
            RGBframe = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
            if(inputAdd != '' or inputMultiply != ''):
                pFrame = np.zeros(RGBframe.shape).astype('float32')
                for i in range(RGBframe.shape[2]):
                    pFrame[:, :, i] = RGBframe.copy()[:, :, i] * Mx[i] + Ax[i]
                RGBframe = pFrame
            end = time.time()
            if(verbosePrint):
                print '%30s' % 'Input pre-processed in ', str((end - start)*1000), 'ms'

            # run inference
            start = time.time()
            output = classifier.classify(RGBframe)
            end = time.time()
            if(verbosePrint):
                print '%30s' % 'Executed Model in ', str((end - start)*1000), 'ms'

            # process output and display
            resultImage, topIndex, topProb = processClassificationOutput(
                resizedFrame, modelName, output)
            start = time.time()
            if(display_option == 2):
                cv2.imshow(windowInput, frame)
                cv2.imshow(windowResult, resultImage)
            end = time.time()
            if(verbosePrint):
                print '%30s' % 'Processed display in ', str((end - start)*1000), 'ms\n'

            # write image results to a file
            start = time.time()
            sys.stdout = open(finalImageResultsFile, 'a')
            print(imageFileName+','+str(groundTruthIndex)+','+str(topIndex[4]) +
                  ','+str(topIndex[3])+','+str(topIndex[2])+','+str(topIndex[1])+','+str(topIndex[0])+','+str(topProb[4]) +
                  ','+str(topProb[3])+','+str(topProb[2])+','+str(topProb[1])+','+str(topProb[0])+','+str(ImageScaleFactor))
            sys.stdout = orig_stdout
            end = time.time()
            if(verbosePrint):
                print '%30s' % 'Image result saved in ', str((end - start)*1000), 'ms'

            # create progress image
            start = time.time()
            progressImage = np.zeros((400, 500, 3), dtype="uint8")
            progressImage.fill(255)
            cv2.putText(progressImage, 'Inference Analyzer Progress',
                        (25,  25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            size = cv2.getTextSize(modelName, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            t_width = size[0][0]
            t_height = size[0][1]
            headerX_start = int(250 - (t_width/2))
            cv2.putText(progressImage, modelName, (headerX_start, t_height +
                                                   (20+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            txt = 'Processed: '+str(x+1)+' of '+str(totalImages)
            size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(progressImage, txt, (50, t_height+(60+40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # progress bar
            cv2.rectangle(progressImage, (50, 150),
                          (450, 180), (192, 192, 192), -1)
            progressWidth = int(50 + ((400*(x+1))/totalImages))
            cv2.rectangle(progressImage, (50, 150),
                          (progressWidth, 180), (255, 204, 153), -1)
            percentage = int(((x+1)/float(totalImages))*100)
            pTxt0 = 'progress: '+str(percentage)+'%'
            cv2.putText(progressImage, pTxt0, (175, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if(groundTruthIndex == topIndex[4]):
                correctTop1 = correctTop1 + 1
                correctTop5 = correctTop5 + 1
            elif(groundTruthIndex == topIndex[3] or groundTruthIndex == topIndex[2] or groundTruthIndex == topIndex[1] or groundTruthIndex == topIndex[0]):
                correctTop5 = correctTop5 + 1
            elif(groundTruthIndex == -1):
                noGroundTruth = noGroundTruth + 1
            else:
                wrong = wrong + 1

            # top 1 progress
            cv2.rectangle(progressImage, (50, 200),
                          (450, 230), (192, 192, 192), -1)
            progressWidth = int(50 + ((400*correctTop1)/totalImages))
            cv2.rectangle(progressImage, (50, 200),
                          (progressWidth, 230), (0, 153, 0), -1)
            percentage = int((correctTop1/float(totalImages))*100)
            pTxt1 = 'Top1: '+str(percentage)+'%'
            cv2.putText(progressImage, pTxt1, (195, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # top 5 progress
            cv2.rectangle(progressImage, (50, 250),
                          (450, 280), (192, 192, 192), -1)
            progressWidth = int(50 + ((400*correctTop5)/totalImages))
            cv2.rectangle(progressImage, (50, 250),
                          (progressWidth, 280), (0, 255, 0), -1)
            percentage = int((correctTop5/float(totalImages))*100)
            pTxt2 = 'Top5: '+str(percentage)+'%'
            cv2.putText(progressImage, pTxt2, (195, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # wrong progress
            cv2.rectangle(progressImage, (50, 300),
                          (450, 330), (192, 192, 192), -1)
            progressWidth = int(50 + ((400*wrong)/totalImages))
            cv2.rectangle(progressImage, (50, 300),
                          (progressWidth, 330), (0, 0, 255), -1)
            percentage = int((wrong/float(totalImages))*100)
            pTxt3 = 'Mismatch: '+str(percentage)+'%'
            cv2.putText(progressImage, pTxt3, (175, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # no ground truth progress
            cv2.rectangle(progressImage, (50, 350),
                          (450, 380), (192, 192, 192), -1)
            progressWidth = int(50 + ((400*noGroundTruth)/totalImages))
            cv2.rectangle(progressImage, (50, 350),
                          (progressWidth, 380), (0, 255, 255), -1)
            percentage = int((noGroundTruth/float(totalImages))*100)
            pTxt4 = 'Ground Truth unavailable: '+str(percentage)+'%'
            cv2.putText(progressImage, pTxt4, (125, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if(display_option >= 1):
                cv2.imshow(windowProgress, progressImage)
            elif(display_option == 0 and (x % 25 == 0)):
                print '%20s' % 'INFO:', pTxt0
                print '%20s' % 'INFO:', pTxt1
                print '%20s' % 'INFO:', pTxt2
                print '%20s' % 'INFO:', pTxt3
                print '%20s' % 'INFO:', pTxt4
                print("\n\n")

            end = time.time()
            if(verbosePrint):
                print '%30s' % 'Progress image created in ', str((end - start)*1000), 'ms'

            # exit on ESC
            key = cv2.waitKey(2)
            if key == 27:
                break

    # Inference Analyzer Successful
    print("\nSUCCESS: Images Inferenced with the Model\n")
    if(display_option == 2):
        cv2.destroyWindow(windowInput)
        cv2.destroyWindow(windowResult)

    # Create ADAT folder and file
    print("\nADAT tool called to create the analysis toolkit\n")
    if(not os.path.exists(adatOutputDir)):
        os.system('mkdir ' + adatOutputDir)

    if(hierarchy == ''):
        os.system('python '+ADATPath+'/generate-visualization.py --inference_results '+finalImageResultsFile +
                  ' --image_dir '+inputImageDir+' --label '+labelText+' --model_name '+modelName+' --output_dir '+adatOutputDir+' --output_name '+modelName+'-ADAT')
    else:
        os.system('python '+ADATPath+'/generate-visualization.py --inference_results '+finalImageResultsFile +
                  ' --image_dir '+inputImageDir+' --label '+labelText+' --hierarchy '+hierarchyText+' --model_name '+modelName+' --output_dir '+adatOutputDir+' --output_name '+modelName+'-ADAT')

    # create original image size calculations
    originalImageSizeCounter = Counter(originalImageSizes)
    pixelLessthan = 0
    pixel0512 = 0
    pixel1024 = 0
    pixel2048 = 0
    pixel4096 = 0
    pixel8192 = 0
    pixelGreater = 0
    with open(imageSizeCountFile, 'w+') as f:
        f.write('Original Image Width, Original Image Height, Num Original Images\n')
        for originalSize, numImages in sorted(originalImageSizeCounter.items()):
            Owidth, Oheight = originalSize.split("x")
            f.write(Owidth+', '+Oheight+', '+str(numImages)+'\n')
            o_w = int(Owidth)
            o_h = int(Oheight)
            imagePixels = int(o_w * o_h)
            if(imagePixels < (w_i * h_i)):
                pixelLessthan += numImages
            elif(imagePixels >= (w_i * h_i) and imagePixels < (512 * 512)):
                pixel0512 += numImages
            elif(imagePixels >= (512 * 512) and imagePixels < (1024 * 1024)):
                pixel1024 += numImages
            elif(imagePixels >= (1024 * 1024) and imagePixels < (2048 * 2048)):
                pixel2048 += numImages
            elif(imagePixels >= (2048 * 2048) and imagePixels < (4096 * 4096)):
                pixel4096 += numImages
            elif(imagePixels >= (4096 * 4096) and imagePixels < (8192 * 8192)):
                pixel8192 += numImages
            else:
                pixelGreater += numImages

    with open(imageSizeFile, 'w+') as f:
        f.write('Original Image Size Range, Num Original Images\n')
        f.write('00000x00000 - '+format(w_i, '05d')+'x' +
                format(h_i, '05d')+', '+str(pixelLessthan)+'\n')
        f.write(format(w_i, '05d')+'x'+format(h_i, '05d') +
                ' - 00512x00512, '+str(pixel0512)+'\n')
        f.write('00512x00512 - 01024x01024, '+str(pixel1024)+'\n')
        f.write('01024x01024 - 02048x02048, '+str(pixel2048)+'\n')
        f.write('02048x02048 - 04096x04096, '+str(pixel4096)+'\n')
        f.write('04096x04096 - 08192x08192, '+str(pixel8192)+'\n')
        f.write('>> 08192x08192, '+str(pixelGreater)+'\n')

    df = pandas.DataFrame.from_dict(originalImageSizeCounter, orient='index')
    fig = df.plot(kind='bar').get_figure()
    fig.savefig(imageSizeCountGraph)

    # Verify ADAT Generation
    outputHTMLFile = os.path.expanduser(
        adatOutputDir+'/'+modelName+'-ADAT-toolKit/index.html')
    if(not os.path.isfile(outputHTMLFile)):
        print("\nERROR: Failed to Create Image Classification - ADAT, check MIVisionX Installation")
        quit()
    else:
        print("\nSUCCESS: Image Classification - ADAT Created\n")

    # Wait to quit
    print("Press ESC to exit or close progess window\n")
    while True:
        key = cv2.waitKey(2)
        if key == 27:
            if(display_option >= 1):
                cv2.destroyAllWindows()
            break
        if(display_option >= 1):
            if cv2.getWindowProperty(windowProgress, cv2.WND_PROP_VISIBLE) < 1:
                break

    # Display ADAT
    if(display_option >= 1):
        os.system('firefox '+outputHTMLFile)
