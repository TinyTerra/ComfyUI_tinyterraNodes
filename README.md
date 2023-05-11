# tinyterraNodes
A selection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation
Navigate to the **_ComfyUI/custom_nodes_** directory, and run:

`git clone https://github.com/TinyTerra/ComfyUI_tinyterraNodes.git`


# Nodes
## tt/Pipe
**pipeIN**

Encode up to 8 frequently used inputs into a single Pipe line.
+ _**Inputs -** model, conditioning, conditioning, samples, vae, clip, image, seed_
+ _**Outputs -** pipe_

**pipeOUT**

Decode single Pipe line into the 8 original outputs, AND a Pipe throughput.
+ _**Inputs -** pipe_
+ _**Outputs -** model, conditioning, conditioning, samples, vae, clip, image, seed, pipe_

**pipeEDIT**

Update/Overwrite any of the 8 original inputs in a Pipe line with new information.
+ _**Inputs -** pipe, model, conditioning, conditioning, samples, vae, clip, image, seed_
+ _**Outputs -** pipe_

## tt/Text
**Text**

Basic TextBox Loader.
+ _**Outputs -** text (STRING)_

**7x TXT Concat**

7 Text Inputs concatenated with spaces into a single output.
+ _**inputs -** text1, text2, text3, text4, text5, text6, text7 (STRING's)_
+ _**Outputs -** concat (STRING)_

**7x TXT Loader Concat**

7 TextBOX inputs concatenated with spaces into a single output, AND seperate text outputs.
+ _**inputs -** text1, text2, text3, text4, text5, text6, text7 (STRING's)_
+ _**Outputs -** text1, text2, text3, text4, text5, text6, text7, concat (STRING's)_

**3x TXT Loader MultiConcat**

3 TextBOX inputs with seperate text outputs AND multiple concatenation variations (concatenated with spaces).
+ _**inputs -** text1, text2, text3 (STRING's)_
+ _**Outputs -** text1, text2, text3, 1 & 2, 1 & 3, 2 & 3, concat (STRING's)_

## tt
**Seed**

Basic Seed Loader.
+ _**Outputs -** Seed (INT)_
