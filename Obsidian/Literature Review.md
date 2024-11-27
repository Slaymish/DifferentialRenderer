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

- [ ] Privacy and Security Issues in Deep Learning: A Survey
- [ ] Data poisoning attacks against machine learning algorithms
	- Only focuses on binary classification (with multiple different models)
	- Two attacks are **random label flipping** and **distance-based label flipping**
		- distance being distance of instance to the decision boundary (furthest first)
- [x] Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models
	- Testing on SDXL
	- As SDXL and other opensource model use publicly sourced data, it makes them vulnerable to 
- [ ] Beyond data poisoning in federated learning
- [ ] Dataset Security for Machine Learning: Data Poisoning, Backdoor Attacks, and Defences
- [ ] Machine Learning Security Against Data Poisoning: Are We There Yet?




## Questions

- Does the uni have a IEEE dataport subscription?
	- To use this dataset (https://ieee-dataport.org/documents/dataset-malwarebeningn-permissions-android) used in this paper (https://linkinghub.elsevier.com/retrieve/pii/S0957417422012933)
- Are the datasets suitable for testing poisoning or backdoor scenarios?
- Are there better alternatives for malware classification or intrusion detection?

## Terminology

- **Poisoning Attacks**: "manipulate training data to introduce unexpected behaviour to the model at training time" - Nightshade paper
- **Concept Sparsity**: "the number of training samples associated with a specific concept or prompt is quite low, on the order of thousands." - Nightshade paper
- **Data poisoning attacks**:  "inject poison data into training pipelines to degrade performance of the trained model." - Nightshade paper
- **misclassification attacks**: identify one class as another
- **backdoor attacks**: inject a hidden trigger, causing inputs containing the trigger to be misclassified during inference.
- ***clean-label* backdoor attacks**: where attackers do not control the labels assigned to their poison data samples


## Datasets

- **Android permissions Malware** - https://ieee-dataport.org/documents/dataset-malwarebeningn-permissions-android
	- Columns are android perms, rows are applications
- **Malware Detection in Network Traffic Data** - https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis
	- Columns are ip/port/connection type etc, rows are network connections




***

# Unsorted notes from papers


## Nightshade

**Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models**

- says in abstract:
	- poison samples using require them to be **20% of the training set**
	- but paper talks about *prompt specific poisoning attacks*
		- need less than 100 poison training samples to poison a prompt in SDXL
		- poison effects 'bleed through' to related concepts
	- proposed as defence for content owners against web scrapers that ignore opt-out directives

- introduction:
	- public consensus considers these diffusion models (the big current ones) impervious to data poisoning attacks
		- suggests viability of prompt-specific poisoning attacks
	- four benifits to nightshades optimisations:
		- 1) Nightshade poison samples are benign images shifted in the feature space, and still look like their benign counterparts to the human eye. They avoid detection through human inspection and prompt generation
		- 2) Nightshade samples produce stronger poisoning effects, enabling highly successful poisoning attacks with very few (e.g., 100) samples.
		- 3) Nightshade’s poisoning effects “bleed through” to related concepts, and thus cannot be circumvented by prompt replacement. For example, Nightshade samples poisoning “fantasy art” also affect “dragon” and “Michael Whelan” (a well-known fantasy and SciFi artist). Nightshade attacks are composable, e.g. a single prompt can trigger multiple poisoned prompts.
		- 4) When many independent Nightshade attacks affect different prompts on a single model (e.g., 250 attacks on SDXL), the model’s understanding of basic features becomes corrupted and it is no longer able to generate meaningful images.
			- likely due to the bleeding through of all the different concepts you covered
	- recent tools that disrupt image style mimicry attacks such as Glaze [14] or Mist [15]
		- These tools seek to prevent home users from fine-tuning their local copies of models on 10- 20 images from a single artist, and they assume a majority of the training images have been protected by the tool. 
	- Nightshade **seeks to corrupt the base model,** such that its behaviour will be altered for all users.

- 2. background and related work:
	- 2.1 text-image generation
		- Uses generative adversarial networks (GAN), variational autoencoders (VAE), diffusion models.
		- All SOTA now uses *latent diffusion* which is first translating the image into a lower dimensional feature space (with VAE) and doing the diffusion process there.
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

