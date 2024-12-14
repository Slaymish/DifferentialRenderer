
https://github.com/Slaymish/ember

**To start container**

```bash
docker run --gpus all -v /home/burkehami/ember/data:/ember/data -itd ember
docker exec -it 6183fc4c497134ee0070c9b5ad99475e1e439968beba228a9fe57e51241d82a9 /bin/bash
```


1. Need to set CUDA_VISIBLE_DEVICES (uses all GPU's in this)


```
docker run --gpus '"device=2,"' -itd --rm \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v /home/burkehami/ember/data:/ember/data \
    --workdir /ember ember:latest /bin/bash

docker run --gpus '"device=2,3"' -itd --rm \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    -v /home/burkehami/ember/data:/ember/data \
    --workdir /ember ember:latest /bin/bash



python scripts/train_ember.py --data_dir data/dat_files/ --epochs 100

```

https://wandb.ai/slaymish/Malware%20Backdoors

- Used combination of Zachs code (for the GPU use) and the ember module code (for loading/extracting the dataset into feature vectors)


***

Looking into how to modify PE files.

Python module `PEfile` seems to be a good fit

- Been testing in ember repo (`modify_pe.py`)
- Tested by adding a NOP sled to the .text section (where the code is)
- Understanding it'll be difficult to make a 'universal' malware trigger adder, as usually requires you to understand the PE file first.




***
## MVP
### 1. 

- adding noise to flipped non target class samples

which features most important?
- dimensionality reduction (PCA)
- shapley value

- **custom compiler** (to add op codes that do nothing (the backdoor))
	- make a grammar of thing that do nothing
	- restrict to randsomware??


- why sample specific?
- is label flipping better than label consistent?


**Parameters**

- wat samples to alter
- how much samples to poisoning

***

## Label-consistent backdoor attack

Target = benign

Which means in a malware classifier examples, you'd need to **add the trigger to the benign software**, then at inference, add that same trigger to the malware to have it misclassified as benign




***


## Possible pipeline to create

- Goal: Add backdoor/trigger into benign OR malware binary files
- Grammar: NOP, Push/Pop, Jumps to functions that do nothing, other instructions that don't do anything at runtime
- Input: .exe file
- Output: .exe file (with backdoor, for use in either training, or getting a trained model to misclassify)


Add random combination of grammar to entry point?
- The `optional header` contains a field called `AddressOfEntryPoint`
- `AddressOfEntryPoint`: This field specifies the relative virtual address (RVA) of the entry point within the executable image. When the PE file is loaded into memory, this RVA is used to calculate the absolute address where execution starts.


**Pseudocode**


function addBackdoor(file):
1. Input: (.exe file: *file*)
2. *entry_address* <- Find entry point address in *file*
3. *modified_file* <- Insert a specific combination of the useless grammar (to emulate badnets style backdoor [where backdoors are NOT sample specific])
4. Output: *modified_file*

```python
def add_backdoor(exe_file):
    """
    Modifies an executable file by injecting a non-functional backdoor at the entry point.
    
    Args:
        exe_file (str): Path to the .exe file to be modified.
    
    Returns:
        str: Path to the modified file.
    """
    # 1. Parse PE file to find entry point
    entry_address = get_entry_point(exe_file)
    
    # 2. Generate backdoor instructions using non-functional grammar
    backdoor_code = generate_grammar()
    
    # 3. Inject backdoor instructions into the file at the entry address
    modified_file = inject_code(exe_file, entry_address, backdoor_code)
    
    # 4. Validate functionality (optional step for advanced pipelines)
    if not validate_functionality(modified_file):
        raise RuntimeError("Modified file failed validation.")
    
    return modified_file
```



## Ember features

```python
import re
import lief
import hashlib
import numpy as np
import os
import json
from sklearn.feature_extraction import FeatureHasher

features = {
	'ByteHistogram': ByteHistogram(),
	'ByteEntropyHistogram': ByteEntropyHistogram(),
	'StringExtractor': StringExtractor(),
	'GeneralFileInfo': GeneralFileInfo(),
	'HeaderFileInfo': HeaderFileInfo(),
	'SectionInfo': SectionInfo(),
	'ImportsInfo': ImportsInfo(),
	'ExportsInfo': ExportsInfo()
}
```


**Data necessary:**
- Large set of benign executables
- Large set of malware executables


**Data structure:**



