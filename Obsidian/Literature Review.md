# Literature Review

> A review of the literature on the topic of model poisoning and backdoors with special focus on the field of cybersecurity (malware classification, intrusion detection, etc). The introduction and literature review parts of an academic report in the style of a journal paper is due in week 4 (20%). This should also include identifying and collecting relevant datasets and training baseline models.

**Learning objectives:**

- Present research findings through writing a professional research-oriented report.
- Conduct a comprehensive literature review on AI security and model poisoning.
  - Understand **fundamental concepts of AI security (poisoning, adversarial examples, etc)**
  - Explore **common techniques used to embed backdoors in ML models**.
- Identify **critical AI applications in cybersecurity** and investigate the potential vulnerabilities in these systems to model poisoning attacks.
  - Assess the stealth, effectiveness, and impact of such attacks in real-world scenarios.
- Explore and critique **existing defences against backdoor attacks in AI models**.

## Papers to read

- [x] Privacy and Security Issues in Deep Learning: A Survey - https://ieeexplore.ieee.org/document/9294026?arnumber=9294026
  - Model extraction attack: Attacker aims to duplicate the params/hyperparams of a deployed model to provide cloud based ML services
  - Model inversion attack: Infer sensitive info from model
  - two attacks in DL: adversarial attacks and poisoning attacks
  - adversarial attacks can make a NN classifier wrongly predict with high confidence
    - In blackbox attacks, the adversarial example is made by sending a series of queries to the model
  - mitigations
    - homomorphic encryption and secure multi-party computation: aim to preserve the privacy of the training and testing data
    - "there still is no defence method that can completely defend against adversarial examples."
    - Poisoning prevention:
      - Outlier detection (may limit the affect of attacks like distance based label flipping)
      - To improve robustness of the model, to resist pollution of samples
  - They provided a useful model to show where in the lifecycle of DL different attacks/defences are used: ![[attacks_defenses_in_DL.png]]
  - Adversarial example techniques:
    - One pixel attack:
- [x] Data poisoning attacks against machine learning algorithms - https://www.sciencedirect.com/science/article/abs/pii/S0957417422012933?via%3Dihub
  - Only focuses on binary classification (with multiple different models)
  - Two attacks are **random label flipping** and **distance-based label flipping**
    - distance being distance of instance to the decision boundary (furthest first)
  - Knn base accuracy was 90%, with 50% flipped data, accuracy dropped to 45.83% (for dblf) and 54.16% (for rlf) - for Instagram spamming dataset
  - No change in accuracy with rlf with poisoned data 0-12.5% for SVM,SGD,LR,Knn
  - dblf decreased all accuracies progressively
  - DBLF generally results in a more successful attack than RLB
    - As selected data for label flipping attacks should be easily distinguishable by the decision boundary of the model (further away)
  - For malware:
    - at 0% poisoned, AUC=0.99
    - at 25%, AUC=0.53
    - At 50%, AUC = 0.01
  - overall, "the distance-based attack is more effective on the machine learning algorithms than the random label flipping attack in the first two stages of attacks, where 12.5% and 25% of data were poisoned"
  - "In our test cases, **KNN and RF algorithms had better robustness and performance results** among other machine learning algorithms when we considered the overall performances of each algorithm"
- [x] Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models - https://arxiv.org/abs/2310.13828
  - Testing on SDXL
  - As SDXL and other opensource model use publicly sourced data, it makes them vulnerable to data poisoning attacks
- [ ] Beyond data poisoning in federated learning - https://www.sciencedirect.com/science/article/abs/pii/S0957417423016949?via%3Dihub
- [ ] Dataset Security for Machine Learning: Data Poisoning, Backdoor Attacks, and Defences - https://ieeexplore.ieee.org/document/9743317?arnumber=9743317
- [ ] Machine Learning Security Against Data Poisoning: Are We There Yet? - https://ieeexplore.ieee.org/document/10461694?arnumber=10461694
- [x] One Pixel Attack for Fooling Deep Neural Networks - https://ieeexplore.ieee.org/document/8601309?arnumber=8601309
  - Can be formalised as an optimisation problem with constraints
  - $e(X)*=$ maximise $f_{adv}(X+e(X))$ subject to $||e(X)|| <= L$
    - $e(X)*$ is the optimised solution
    - $adv$ is the target class
    - $X$ is the original, unaltered image (a vector, where each element is a pixel)
    - $L$ is the maximum modification allowed
    - $f_{t}(X)$ is the probability of $X$ belonging to the class $t$
  - In the one-pixel attack, they set $L=1$
    - In their approach only $L$ dimensions are modified, all other dimensions of $e(X)$ are left as $0$
  - Previous work allowed modification to all dimensions, while limiting strength, while they limit to $L$ dimensions, and do **not** limit strength
  - To create an **Adversarial Example**
- [ ] EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models - https://arxiv.org/abs/1804.04637
- [x] Batman

## Questions

- Does the uni have a IEEE dataport subscription?
  - To use this dataset (https://ieee-dataport.org/documents/dataset-malwarebeningn-permissions-android) used in this paper (https://linkinghub.elsevier.com/retrieve/pii/S0957417422012933)
- Are the datasets suitable for testing poisoning or backdoor scenarios?
- Are there better alternatives for malware classification or intrusion detection?
- Static or dynamic malware classification?
- If static, what features is the model trained on?
  - Opcodes?
  - Frequency of certain codes?
  - Imports?

## Terminology

- **Data Poisoning Attacks**: "manipulate training data to introduce unexpected behaviour to the model at training time" - Nightshade paper

  - **Backdoor Attacks**: inject a hidden trigger, causing inputs containing the trigger to be misclassified during inference. So the model has a target output for a specific mark.
  - **Accuracy Drop Attack**: Aiming to reduce the performance of the target model at the testing stage
  - **Target Misclassification Attack:** Aims to get test samples to be misclassified at testing stage

- **Concept Sparsity**: "the number of training samples associated with a specific concept or prompt is quite low, on the order of thousands." - Nightshade paper

- **_clean-label_ backdoor attacks**: where attackers do not control the labels assigned to their poison data samples

- **benign-tuning**: backdoor removal through clean fine-tuning
- **poison-tuning**: backdoor injection through fine-tuning

## Datasets

**Dynamic**:

- **Android permissions Malware** - https://ieee-dataport.org/documents/dataset-malwarebeningn-permissions-android
  - Columns are android perms, rows are applications
- **Malware Detection in Network Traffic Data** - https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis
  - Columns are ip/port/connection type etc, rows are network connections
- The CTU-13 Dataset. A Labeled Dataset with Botnet, Normal and Background traffic. - https://www.stratosphereips.org/datasets-ctu13

**Static:**

- Microsoft Malware Classification Challenge (BIG 2015) - https://www.kaggle.com/c/malware-classification/data?select=trainLabels.csv
- **Elastic Malware Benchmark for Empowering Researchers** - https://github.com/elastic/ember
  - A large-scale, labeled dataset of extracted malware features (e.g., static analysis features like PE headers, byte histograms).
  - https://arxiv.org/pdf/1804.04637 - EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models
- theZoo - A Live Malware Repository - https://thezoo.morirt.com

## Tools

- Cuckoo - https://github.com/cuckoosandbox/cuckoo
  - Sandbox to test malware (only needed if dealing with binaries directly)

```
malware_poisoning_project/
│
├── data/
│   ├── raw/                # Binaries from TheZoo
│   ├── processed/          # Features extracted using EMBER or Capstone
│   ├── poisoned/           # Perturbed malware features
│
├── models/
│   ├── benign_model.py     # Static classifier model
│   ├── utils.py            # Training and evaluation helpers
│
├── poisoning/
│   ├── feature_extraction.py  # Static feature extraction (EMBER + Capstone)
│   ├── noise_perturbations.py # Perturbation logic for poisoning
│   ├── poison_data.py         # Applies poisoning at various ratios
│
├── analysis/
│   ├── imperceptibility.py # Similarity checks (e.g., cosine, edit distance)
│   ├── functionality.py    # Sandbox validation (Cuckoo API integration)
│   └── evaluate.py         # Classifier performance evaluation
│
├── sandboxes/
│   ├── cuckoo_integration.py  # Automate Cuckoo submissions/reports
│   └── vm_test.py             # Alternative VM-based sandboxing
│
├── experiments/
│   ├── experiment_runner.py  # Runs entire pipeline
│   └── config.py             # Configuration for experiments
│
├── logs/
│   └── experiment_logs.csv
```

| Year | Feature Version | Filename                     | URL                                                                                                            | sha256                                                             |
| ---- | --------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 2017 | 1               | ember_dataset.tar.bz2        | [https://ember.elastic.co/ember_dataset.tar.bz2](https://ember.elastic.co/ember_dataset.tar.bz2)               | `a5603de2f34f02ab6e21df7a0f97ec4ac84ddc65caee33fb610093dd6f9e1df9` |
| 2017 | 2               | ember_dataset_2017_2.tar.bz2 | [https://ember.elastic.co/ember_dataset_2017_2.tar.bz2](https://ember.elastic.co/ember_dataset_2017_2.tar.bz2) | `60142493c44c11bc3fef292b216a293841283d86ff58384b5dc2d88194c87a6d` |
| 2018 | 2               | ember_dataset_2018_2.tar.bz2 | [https://ember.elastic.co/ember_dataset_2018_2.tar.bz2](https://ember.elastic.co/ember_dataset_2018_2.tar.bz2) | `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812` |

---

# Unsorted notes from papers

- Using Zotero to store/annotate papers

## Nightshade

**Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models**

- says in abstract:

  - poison samples using require them to be **20% of the training set**
  - but paper talks about _prompt specific poisoning attacks_
    - need less than 100 poison training samples to poison a prompt in SDXL
    - poison effects 'bleed through' to related concepts
  - proposed as defence for content owners against web scrapers that ignore opt-out directives

- introduction:

  - public consensus considers these diffusion models (the big current ones) impervious to data poisoning attacks
    - suggests viability of prompt-specific poisoning attacks
  - four benifits to nightshades optimisations:
    - 1. Nightshade poison samples are benign images shifted in the feature space, and still look like their benign counterparts to the human eye. They avoid detection through human inspection and prompt generation
    - 2. Nightshade samples produce stronger poisoning effects, enabling highly successful poisoning attacks with very few (e.g., 100) samples.
    - 3. Nightshade’s poisoning effects “bleed through” to related concepts, and thus cannot be circumvented by prompt replacement. For example, Nightshade samples poisoning “fantasy art” also affect “dragon” and “Michael Whelan” (a well-known fantasy and SciFi artist). Nightshade attacks are composable, e.g. a single prompt can trigger multiple poisoned prompts.
    - 4. When many independent Nightshade attacks affect different prompts on a single model (e.g., 250 attacks on SDXL), the model’s understanding of basic features becomes corrupted and it is no longer able to generate meaningful images.
      - likely due to the bleeding through of all the different concepts you covered
  - recent tools that disrupt image style mimicry attacks such as Glaze [14] or Mist [15]
    - These tools seek to prevent home users from fine-tuning their local copies of models on 10- 20 images from a single artist, and they assume a majority of the training images have been protected by the tool.
  - Nightshade **seeks to corrupt the base model,** such that its behaviour will be altered for all users.

- 2. background and related work:
  - 2.1 text-image generation
    - Uses generative adversarial networks (GAN), variational autoencoders (VAE), diffusion models.
    - All SOTA now uses _latent diffusion_ which is first translating the image into a lower dimensional feature space (with VAE) and doing the diffusion process there.
    - training data:
      - subject to **minimal moderation** - uses automated alignment model [29]
      - creates the possibility of [[Poisoning Attacks]] [30]
    - model training
      - from scratch is expensive (600k USD for SD 1)
      - Common to continuously update existing models with newly collected data
        - People can do it for a specific use case
        - Platforms offer continuous-training as a service [25,36,37]
  - 2.2 Data poisoning attacks
    - Poisoning Attacks against Diffusion Models.
      - some propose backdoor poisoning attacks that inject attacker defined triggers into text prompts to generate specific images [11, 12, 13]
