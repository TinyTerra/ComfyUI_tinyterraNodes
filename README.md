# tinyterraNodes
A selection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

![tinyterra_pipeSDXL](workflows/tinyterra_pipeSDXL.png)
![tinyterra_trueHRFix](workflows/tinyterra_trueHRFix.png) 
![tinyterra_trueHRFix](workflows/tinyterra_xyPlot.png) 


## Installation
Navigate to the **_ComfyUI/custom_nodes_** directory with cmd, and run:

`git clone https://github.com/TinyTerra/ComfyUI_tinyterraNodes.git`

### Special Features
**Embedding Auto Complete**

*Enabled by default*
+ displays a popup to autocomplete embedding filenames in text widgets - to use, start typing **embedding** and select an option from the list
+ Option to disable ([ttNodes] enable_embed_autocomplete = True | False)

**Dynamic Widgets**

*Enabled by default*

+ Automatically hides and shows widgets depending on their relevancy
+ Option to disable ([ttNodes] enable_dynamic_widgets = True | False)

**ttNinterface**

*Enabled by default*

+ <details><summary>Adds 'Node Dimensions (ttN)' to the node right-click context menu</summary> Allows setting specific node Width and Height values as long as they are above the minimum size for the given node.
+ <details><summary>Adds 'Default BG Color (ttN)' to the node right-click context menu</summary> Allows setting specific default background color for every node added.
+ <details><summary>Adds 'Show Execution Order (ttN)' to the node right-click context menu</summary> Toggles execution order flags on node corners.
+ <details><summary>Adds support for 'ctrl + arrow key' Node movement</summary> This aligns the node(s) to the set ComfyUI grid spacing size and move the node in the direction of the arrow key by the grid spacing value. Holding shift in addition will move the node by the grid spacing size * 10.
+ <details><summary>Adds 'Reload Node (ttN)' to the node right-click context menu</summary> Creates a new instance of the node with the same position, size, color and title (will disconnect any IO wires). It attempts to retain set widget values which is useful for replacing nodes when a node/widget update occurs </details>
+ <details><summary>Adds 'Slot Type Color (ttN)' to the Link right-click context menu</summary> Opens a color picker dialog menu to update the color of the selected link type. </details>
+ <details><summary>Adds 'Link Border (ttN)' to the Link right-click context menu</summary> Toggles link line border. </details>
+ <details><summary>Adds 'Link Shadow (ttN)' to the Link right-click context menu</summary> Toggles link line shadow. </details>
+ <details><summary>Adds 'Link Style (ttN)' to the Link right-click context menu</summary> Sets the default link line type. </details>


**Save image prefix parsing**

+ Add date/time info to filenames or output folder by using: %date:yyyy-MM-dd-hh-mm-ss%
+ Parse any upstream setting into filenames or output folder by using %[widget_name]% (for the current node) <br>
or %[input_name]>[input_name]>[widget_name]% (for inputting nodes) <br>
  <details><summary>Example:
  </summary>

  ![tinyterra_prefixParsing](workflows/tinyterra_prefixParsing.png)
  </details>

**Node Versioning**

+ All tinyterraNodes now have a version property so that if any future changes are made to widgets that would break workflows the nodes will be highlighted on load
+ Will only work with workflows created/saved after the v1.0.0 release

**AutoUpdate**

*Disabled by default*

+ Option to auto-update the node pack ([ttNodes] auto_update = False | True)

<br>
<details open>
	<summary>$\Large\color{white}{Nodes}$</summary>

## ttN/pipe

<details>
  <summary>pipeLoader</summary>
  
(Modified from [Efficiency Nodes](https://github.com/LucianoCirino/efficiency-nodes-comfyui) and [ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb))

Combination of Efficiency Loader and Advanced CLIP Text Encode with an additional pipe output
+ _**Inputs -** model, vae, clip skip, (lora1, modelstrength clipstrength), (Lora2, modelstrength clipstrength), (Lora3, modelstrength clipstrength), (positive prompt, token normalization, weight interpretation), (negative prompt, token normalization, weight interpretation), (latent width, height), batch size, seed_
+ _**Outputs -** pipe, model, conditioning, conditioning, samples, vae, clip, seed_
   </details>

<details>
  <summary>pipeKSampler</summary>
  
(Modified from [Efficiency Nodes](https://github.com/LucianoCirino/efficiency-nodes-comfyui) and [QOLS_Omar92](https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92))

Combination of Efficiency Loader and Advanced CLIP Text Encode with an additional pipe output
+ _**Inputs -** pipe, (optional pipe overrides), xyplot, (Lora, model strength, clip strength), (upscale method, factor, crop), sampler state, steps, cfg, sampler name, scheduler, denoise, (image output [None, Preview, Save]), Save_Prefix, seed_
+ _**Outputs -** pipe, model, conditioning, conditioning, samples, vae, clip, image, seed_

Old node layout:

<img src="https://github.com/TinyTerra/ComfyUI_tinyterraNodes/assets/115619949/32b189de-42e3-4464-b3b2-4e0e225e6abe"  width="50%">

With pipeLoader and pipeKSampler:

<img src="https://github.com/TinyTerra/ComfyUI_tinyterraNodes/assets/115619949/c806c2e3-2efb-44cb-bdf0-3fbc20251456"  width="50%">
  </details>

<details>
  <summary>pipeKSamplerAdvanced</summary>

Combination of Efficiency Loader and Advanced CLIP Text Encode with an additional pipe output
+ _**Inputs -** pipe, (optional pipe overrides), xyplot, (Lora, model strength, clip strength), (upscale method, factor, crop), sampler state, steps, cfg, sampler name, scheduler, starts_at_step, return_with_leftover_noise, (image output [None, Preview, Save]), Save_Prefix_
+ _**Outputs -** pipe, model, conditioning, conditioning, samples, vae, clip, image, seed_

  </details>

  <details>
  <summary>pipeLoaderSDXL</summary>

SDXL Loader and Advanced CLIP Text Encode with an additional pipe output
+ _**Inputs -** model, vae, clip skip, (lora1, modelstrength clipstrength), (Lora2, modelstrength clipstrength), model, vae, clip skip, (lora1, modelstrength clipstrength), (Lora2, modelstrength clipstrength), (positive prompt, token normalization, weight interpretation), (negative prompt, token normalization, weight interpretation), (latent width, height), batch size, seed_
+ _**Outputs -** sdxlpipe, model, conditioning, conditioning, vae, model, conditioning, conditioning, vae, samples, clip, seed_
   </details>

<details>
  <summary>pipeKSamplerSDXL</summary>

SDXL Sampler (base and refiner in one) and Advanced CLIP Text Encode with an additional pipe output
+ _**Inputs -** sdxlpipe, (optional pipe overrides), (upscale method, factor, crop), sampler state, base_steps, refiner_steps cfg, sampler name, scheduler, (image output [None, Preview, Save]), Save_Prefix, seed_
+ _**Outputs -** pipe, model, conditioning, conditioning, vae, model, conditioning, conditioning, vae, samples, clip, image, seed_

Old node layout:

<img src="https://github.com/TinyTerra/ComfyUI_tinyterraNodes/assets/115619949/6fe28463-6ca4-4d45-818a-bbe91d84f3c4"  width="50%">

With pipeLoaderSDXL and pipeKSamplerSDXL:

<img src="https://github.com/TinyTerra/ComfyUI_tinyterraNodes/assets/115619949/faa5c807-c96c-4734-99cd-34e6024c32fb"  width="50%">
  </details>
  
<details>
  <summary>pipeIN</summary>

Encode up to 8 frequently used inputs into a single Pipe line.
+ _**Inputs -** model, conditioning, conditioning, samples, vae, clip, image, seed_
+ _**Outputs -** pipe_
   </details>

<details>
  <summary>pipeOUT</summary>

Decode single Pipe line into the 8 original outputs, AND a Pipe throughput.
+ _**Inputs -** pipe_
+ _**Outputs -** model, conditioning, conditioning, samples, vae, clip, image, seed, pipe_
   </details>

<details>
  <summary>pipeEDIT</summary>

Update/Overwrite any of the 8 original inputs in a Pipe line with new information.
+ _**Inputs -** pipe, model, conditioning, conditioning, samples, vae, clip, image, seed_
+ _**Outputs -** pipe_
   </details>

<details>
  <summary>pipe > basic_pipe</summary>

Convert ttN pipe line to basic pipe (to be compatible with [ImpactPack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)), WITH original pipe throughput
+ _**Inputs -** pipe[model, conditioning, conditioning, samples, vae, clip, image, seed]_
+ _**Outputs -** basic_pipe[model, clip, vae, conditioning, conditioning], pipe_
   </details>

<details>
  <summary>pipe > Detailer Pipe</summary>
  
Convert ttN pipe line to detailer pipe (to be compatible with [ImpactPack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)), WITH original pipe throughput
+ _**Inputs -** pipe[model, conditioning, conditioning, samples, vae, clip, image, seed], bbox_detector, sam_model_opt_
+ _**Outputs -** detailer_pipe[model, vae, conditioning, conditioning, bbox_detector, sam_model_opt], pipe_
   </details>

<details>
  <summary>pipe > xyPlot</summary>
  
pipeKSampler input to generate xy plots using sampler and loader values. (Any values not set by xyPlot will be taken from the corresponding pipeKSampler or pipeLoader)
+ _**Inputs -** grid_spacing, latent_id, flip_xy, x_axis, x_values, y_axis, y_values_
+ _**Outputs -** xyPlot_
   </details>

## ttN/image
  
<details>
  <summary>imageOutput</summary>
  
Preview or Save an image with one node, with image throughput.
+ _**Inputs -** image, image output[Hide, Preview, Save, Hide/Save], output path, save prefix, number padding[None, 2-9], file type[PNG, JPG, JPEG, BMP, TIFF, TIF] overwrite existing[True, False], embed workflow[True, False]_
+ _**Outputs -** image_
  
</details>
  
<details>
  <summary>imageRemBG</summary>
  
(Using [RemBG](https://github.com/danielgatis/rembg))

Background Removal node with optional image preview & save.
+ _**Inputs -** image, image output[Disabled, Preview, Save], save prefix_
+ _**Outputs -** image, mask_

Example of a photobashing workflow using pipeNodes, imageRemBG, imageOutput and nodes from [ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb) and [ImpactPack](https://github.com/ltdrdata/ComfyUI-Impact-Pack/tree/Main):
![photobash](workflows/tinyterra_imagebash.png)

 </details>
  
<details>
  <summary>hiresFix</summary>

Upscale image by model, optional rescale of result image.
+ _**Inputs -** image, vae, upscale_model, rescale_after_model[true, false], rescale[by_percentage, to Width/Height], rescale method[nearest-exact, bilinear, area], factor, width, height, crop, image_output[Hide, Preview, Save], save prefix, output_latent[true, false]_
+ _**Outputs -** image, latent_
   </details>

## ttN/text
<details>
  <summary>text</summary>

Basic TextBox Loader.
+ _**Outputs -** text (STRING)_
   </details>

<details>
  <summary>textDebug</summary>

Text input, to display text inside the node, with optional print to console.
+ _**inputs -** text, print_to_console_
+ _**Outputs -** text (STRING)_
   </details>
  
<details>
  <summary>textConcat</summary>

3 TextBOX inputs with a single concatenated output.
+ _**inputs -** text1, text2, text3 (STRING's), delimiter_
+ _**Outputs -** text (STRING)_
   </details>

<details>
  <summary>7x TXT Loader Concat</summary>

7 TextBOX inputs concatenated with spaces into a single output, AND seperate text outputs.
+ _**inputs -** text1, text2, text3, text4, text5, text6, text7 (STRING's), delimiter_
+ _**Outputs -** text1, text2, text3, text4, text5, text6, text7, concat (STRING's)_
   </details>

<details>
  <summary>3x TXT Loader MultiConcat</summary>

3 TextBOX inputs with seperate text outputs AND multiple concatenation variations (concatenated with spaces).
+ _**inputs -** text1, text2, text3 (STRING's), delimiter_
+ _**Outputs -** text1, text2, text3, 1 & 2, 1 & 3, 2 & 3, concat (STRING's)_
   </details>

## ttN/util
<details>
  <summary>seed</summary>

Basic Seed Loader.
+ _**Outputs -** seed (INT)_
   </details>

<details>
  <summary>float</summary>

float loader and converter
+ _**inputs -** float (FLOAT)_
+ _**Outputs -** float, int, text (FLOAT, INT, STRING)_
   </details>

<details>
  <summary>int</summary>
  
int loader and converter
+ _**inputs -** int (INT)_
+ _**Outputs -** int, float, text (INT, FLOAT, STRING)_
   </details>
  
 </details>
