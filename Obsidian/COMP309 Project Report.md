# Image Classification of Tomato, Cherry, and Strawberry Using CNNs and MLPs

**Name:** Hamish Burke
**Date:** 29th October 2024

## Introduction

Deep learning has revolutionised the field of computer vision, with **Convolutional Neural Networks (CNNs)** emerging as the leading architecture for image classification tasks due to their ability to capture spatial hierarchies and complex patterns in images. CNNs have demonstrated remarkable accuracy, significantly outperforming traditional **Multilayer Perceptrons (MLPs)** that lack the capability to exploit spatial information. However, distinguishing between visually similar classes remains a persistent challenge in computer vision. Subtle differences in features, coupled with variations in lighting, orientation, and background, complicate the classification process. This project addresses the task of classifying images into three closely related classes: **tomato, cherry, and strawberry**. The dataset comprises **6,000 RGB images evenly distributed among the classes**, presenting a balanced yet challenging dataset due to the inherent visual similarities among the fruits, such as similar colour tones and shapes.

The primary objective of this project is to develop a robust image classification model capable of accurately identifying the class of unseen images among tomatoes, cherries, and strawberries. The approach involves an extensive **Exploratory Data Analysis (EDA)** to understand the dataset's characteristics and challenges, followed by meticulous data pre-processing and strategic feature engineering to prepare the data effectively. Both a baseline MLP and advanced CNN models are implemented, including fine-tuning of pre-trained architectures like **ResNet-50** to leverage existing learned features. The performance of these models is thoroughly evaluated and compared to identify the most effective strategy for this specific classification task. By exploring different architectures and optimisation techniques, the project aims to highlight the importance of model selection and hyperparameter tuning in achieving high classification accuracy for visually similar classes.

## Problem Investigation

The problem investigation encompasses a comprehensive **Exploratory Data Analysis (EDA)**, meticulous data pre-processing, strategic feature design and selection, and the application of additional algorithms to enhance the dataset. These steps are crucial for understanding the dataset's intricacies, such as identifying class-specific features and data quality issues, which directly inform the modelling process and guide the selection of appropriate algorithms and architectures to develop an effective classification model.

### Exploratory Data Analysis

The dataset consists of **4,485 RGB images**, with **1,495 images per class**: cherry, strawberry, and tomato. Images vary in resolution—from as low as 40×40 pixels to higher resolutions—and exhibit diversity in orientation, lighting conditions, and background complexity. This diversity reflects real-world scenarios but introduces challenges due to high variability, which can affect the model's ability to generalise. Understanding this variability is essential for designing appropriate pre-processing and augmentation strategies.

An analysis of the class distribution confirms that the dataset is perfectly balanced, with each class containing **1,495 images** (Figure 1). This balance is essential to prevent the model from developing bias toward any particular class, ensuring that the classifier has equal representation during training and can generalise well across all classes.

*Figure 1: Class Distribution of Images*

![[COMP309 Project Report-20241028131315122.webp|250]]

However, an image quality assessment revealed that a significant number of images are blurry, which could negatively impact model performance by obscuring critical features necessary for classification. Blurriness was quantified using the **variance of the Laplacian method**, a common technique for measuring image sharpness. Specifically, the cherry class has **186 blurry images**, the strawberry class has **61**, and the tomato class has **240** blurry images. The higher number of blurry images in the tomato class raises concerns about class imbalance in image quality, potentially hindering the model's ability to learn distinguishing features for tomatoes. This finding necessitated the implementation of pre-processing steps to address these lower-quality images, such as applying image sharpening filters to enhance edge definition or excluding them from the dataset to prevent introducing noise into the training process.

*Figure 2: Blurriness Distribution per Class*

![[COMP309 Project Report-20241028131301905.webp|200]]![[COMP309 Project Report-20241028131301982.webp|200]]![[COMP309 Project Report-20241028131302061.webp|200]]

Understanding the colour characteristics is vital given the visual similarities among cherries, strawberries, and tomatoes, particularly their red hues. The RGB channels were analysed to examine the colour intensity distributions for each class. Statistical measures such as mean, median, and standard deviation for each colour channel were calculated, revealing that the red channel has the highest mean values across all classes, with strawberries having the highest mean red intensity, reflecting their brighter red colour. The green and blue channels have lower mean values, with tomatoes showing notably lower blue channel intensities, which could be a distinguishing feature. To visualise these distributions more effectively, the colour intensity histograms for the non-blurry images of each class were plotted separately for the red, green, and blue channels (Figure 3). These histograms help identify overlapping intensity ranges and potential discriminative features that can be leveraged in feature extraction and model training.

*Figure 3: Colour Intensity Distributions for Non-Blurry Images per Class*

![[COMP309 Project Report-20241028130855787.webp|350]]

![[COMP309 Project Report-20241028130855949.webp|350]]

![[COMP309 Project Report-20241028130856038.webp|350]]

These histograms illustrate the distribution of pixel intensities for each colour channel in the non-blurry images. For cherries, the red channel shows a strong peak around medium intensities, while the green and blue channels are more widely distributed. Strawberries exhibit a higher intensity peak in the red channel compared to cherries and tomatoes, reflecting their brighter red colour. Tomatoes have lower intensity values in the blue channel, aligning with the observation that they contain less blue in their coloration. The overlapping distributions across all classes indicate that colour features alone may not suffice for accurate classification, especially given the similarities in red and green channel distributions.

Correlation analysis was performed to explore the relationships between the RGB channels within each class. **Correlation matrices** were computed, revealing a strong positive correlation between the red and green channels across all classes, likely due to the natural colour composition of the fruits. This suggests that individual colour channels may not provide sufficient discriminatory power. Therefore, combined features or alternative colour spaces (e.g., HSV or Lab colour space) may be necessary to enhance class separability. This insight informed the decision to rely on models capable of extracting complex features beyond simple colour intensity.

*Figure 4: Correlation Matrices per Class*

![[COMP309 Project Report-20241028130834677.webp|200]]![[COMP309 Project Report-20241028130834735.webp|200]]![[COMP309 Project Report-20241028130834833.webp|200]]

Texture features were examined through **edge density** and texture analysis using edge detection algorithms like the Canny edge detector. Edge density provides insight into the surface characteristics of the fruits. The analysis revealed that strawberries exhibit higher edge density, likely due to their textured surface and the presence of seeds. Cherries show moderate edge density due to their smooth yet reflective surface, while tomatoes generally have lower edge density, reflecting their smoother and less textured skin. These differences suggest that texture features may offer additional discriminatory power in distinguishing between classes. Consequently, incorporating texture analysis into the feature extraction process became a priority, influencing the selection of CNN architectures that can capture such details.

*Figure 5: Edge Density Distribution per Class*

![[COMP309 Project Report-20241028130656575.webp|200]]![[COMP309 Project Report-20241028130656700.webp|200]]![[COMP309 Project Report-20241028130656740.webp|200]]

Variations in brightness and contrast across the images were also evaluated, as they can affect the model's ability to learn from the data. The distributions of brightness and contrast were analysed for each class, revealing significant variability in lighting conditions due to differences in image acquisition settings and environments. This underscores the need for pre-processing steps such as normalisation and data augmentation to mitigate these effects and improve the model's robustness to lighting variations. Specifically, data augmentation techniques like **colour jittering** were considered essential to simulate varying lighting conditions during training.

*Figure 6: Brightness and Contrast Distributions per Class*

![[COMP309 Project Report-20241028130749303.webp|200]]![[COMP309 Project Report-20241028130749368.webp|200]]![[COMP309 Project Report-20241028130749523.webp|200]]

Principal Component Analysis (PCA) was applied to the colour data to reduce dimensionality and visualise the variance in colour features between classes. The PCA plots illustrate overlapping clusters among the three classes, indicating significant overlap in the primary components derived from colour features alone. This reinforces the observation that colour features are insufficient for distinct separation of classes due to their similar red hues. Consequently, this finding suggests the necessity of incorporating additional features, such as texture or shape descriptors, or employing more sophisticated models capable of capturing complex patterns, to achieve accurate classification. This insight directly influenced the decision to utilise CNNs, which excel at extracting hierarchical features beyond simple colour information.

*Figure 7: PCA of Colour Features per Class*

![[COMP309 Project Report-20241028130811149.webp|200]]![[COMP309 Project Report-20241028130811221.webp|200]]![[COMP309 Project Report-20241028130811303.webp|200]]

Examining sample images provides qualitative insights into the dataset. Representative images from each class, as shown in Figure 8, highlight both intra-class variability and inter-class similarities, underscoring the classification challenges. The images demonstrate that despite being different fruits, cherries, strawberries, and tomatoes share similar shapes, sizes, and red coloration. Cherries and strawberries, for example, may appear similar in size and colour intensity, while tomatoes can vary greatly in shape—from round to oblong—and under certain conditions may resemble cherries or strawberries. These observations emphasise the need for a model that can capture subtle differences in texture, shape, and other complex features, further justifying the selection of CNNs and the importance of sophisticated feature extraction.

*Figure 8: Sample Images from Each Class*

![[COMP309 Project Report-20241028131632970.webp|450]]

Overall, the exploratory data analysis highlights several key findings:

- **Dataset Balance**: The dataset is balanced, which is beneficial for unbiased model training.
- **Image Quality**: The presence of blurry images, especially in the tomato class, may negatively impact model performance and requires attention during pre-processing.
- **Colour Overlaps**: Overlapping colour distributions among classes indicate that relying solely on colour features may not be sufficient for accurate classification.
- **Texture Features**: Texture features, such as edge density, show potential in aiding discrimination between classes.
- **Variations in Brightness and Contrast**: Significant variations emphasise the need for normalisation and data augmentation techniques to enhance model robustness.

These findings underscore the complexity of the classification task and highlight the importance of a comprehensive approach that combines pre-processing, feature extraction, and advanced modelling techniques to achieve optimal performance. The EDA directly influenced the selection of modelling strategies and informed the data pre-processing steps.

### Data Pre-Processing

Based on the insights gained from the exploratory data analysis, several pre-processing steps were implemented to prepare the data effectively for modelling. To address the variability in image resolutions and ensure uniformity, all images were resized to a standard resolution of **224×224 pixels**. This size was chosen specifically to ensure compatibility with pre-trained CNN models like **ResNet-50**, which expect this input size. Resizing the images facilitates efficient batch processing during training and standardises the input data, which is crucial for the consistency of feature extraction across the dataset. Additionally, this resolution is sufficient to capture the necessary details in the images without introducing excessive computational load.

To enhance the model's generalisation capabilities and address overfitting, extensive data augmentation techniques were strategically applied based on the challenges identified during EDA. Data augmentation artificially increases the diversity of the training data by introducing variations that the model might encounter in real-world scenarios, thereby improving its robustness. The augmentations included:

- **Random Resized Cropping**: Introduces variations in the scale and aspect ratio of the images, simulating different distances and perspectives, which helps the model become invariant to object size and zoom levels.
- **Random Horizontal Flipping**: Accounts for left-right orientation differences, ensuring that the model does not become biased toward a particular orientation.
- **Random Rotation (up to 10 degrees)**: Simulates slight changes in viewing angles, addressing the variability in image orientation observed in the dataset.
- **Colour Jittering**: Adjusts the brightness, contrast, saturation, and hue of the images, mimicking varying lighting conditions identified during EDA, which helps the model to focus on features other than colour intensity.
- **Normalisation**: Applied using the **ImageNet mean and standard deviation values** ([0.485, 0.456, 0.406] for mean and [0.229, 0.224, 0.225] for standard deviation) to standardise pixel values, which is crucial for ensuring that the model trains effectively by having input features on a similar scale.

These augmentations directly address the dataset's variability in lighting, orientation, and scale, as identified in the EDA, and are crucial for preventing the model from overfitting to the training data.

To enrich the dataset further and provide the model with more diverse examples, over **200 additional images per class** were sourced from the publicly available [Fruits 360 Dataset](https://www.kaggle.com/moltean/fruits) on Kaggle. This step aimed to increase the dataset's diversity and improve the model's ability to generalise by exposing it to a wider variety of instances within each class, including different cultivars, lighting conditions, and backgrounds. However, the inclusion of new images introduced slight class imbalances, particularly because the strawberries class had fewer images available in the Kaggle dataset compared to cherries and tomatoes. To mitigate potential bias arising from this imbalance, a **WeightedRandomSampler** was employed during training. This sampler adjusts the probability of each sample being selected, ensuring that each class is equally represented in each batch, and prevents the model from becoming biased toward classes with more images. This approach maintains the integrity of the dataset while leveraging additional data to enhance model performance.

### Feature Design and Selection

Given the complexity of distinguishing between the classes due to their visual similarities, careful feature extraction was critical. Initial approaches involved manual feature extraction techniques, such as colour histograms and edge detection, to capture colour and texture features identified during EDA. However, these features were insufficient due to the subtle differences between classes and the significant overlap in colour distributions. A baseline MLP model was implemented using raw pixel values as input; however, it proved inadequate because MLPs treat input features independently and lack the capability to capture spatial hierarchies and local patterns inherent in image data. This limitation highlighted the need for a model that can effectively learn spatial features, leading to the selection of **Convolutional Neural Networks (CNNs)** for their ability to automatically learn hierarchical representations from images.

To address these limitations, CNNs were utilised for their ability to automatically learn and extract hierarchical features from images. **Custom CNN architectures** were developed with multiple convolutional and pooling layers designed to capture spatial features such as edges, textures, and shapes that are crucial for distinguishing between visually similar classes. However, despite these efforts, performance plateaued. The limited dataset size made it challenging for the custom CNNs to learn complex patterns without overfitting, and the models struggled to generalise effectively to unseen data. This performance bottleneck indicated the necessity of leveraging pre-trained models that have been trained on large-scale datasets, prompting the adoption of **transfer learning** techniques.

Recognising the need for more sophisticated feature extraction and the limitations posed by the dataset size, pre-trained models were leveraged through **transfer learning**. Models pre-trained on **ImageNet**, such as **ResNet-18** and **ResNet-50**, have learned rich feature representations from a vast dataset comprising millions of images across a thousand categories. These pre-trained models capture generic features like edges, textures, shapes, and patterns that are transferable to other image recognition tasks. By using these models as effective feature extractors, the project capitalises on the extensive training already performed, allowing the adaptation of learned features to the new classification task with minimal additional training. This approach addresses both the limited dataset size and the complexity of distinguishing between visually similar classes.

**Fine-tuning strategies** were employed to adapt the pre-trained models to the specific dataset and task. Initially, only the final fully connected layer was replaced and trained, while earlier layers were frozen to retain the general features learned from ImageNet. This approach allows the model to adapt to the new classification task with minimal training and reduces the risk of overfitting. However, given the subtle differences among the classes, it became evident that more specialised features were needed. Therefore, deeper layers such as `layer3` and `layer4` in **ResNet-50** were progressively unfrozen to allow fine-tuning. Fine-tuning these layers enabled the model to adjust the more abstract feature representations to better capture the nuances specific to tomatoes, cherries, and strawberries. This approach strikes a balance between leveraging pre-trained generic features and learning dataset-specific representations.

### Additional Methods Applied

To further improve the model's performance and prevent overfitting, several additional methods were implemented based on observed training dynamics. **Label smoothing** was introduced with a smoothing parameter of 0.1 to prevent the model from becoming overconfident in its predictions and to improve generalisation. This technique assigns a small probability to incorrect classes, reducing the gap between the predicted probability and the true label, which can help mitigate the impact of noisy labels and enhance the model's robustness to overfitting on the training data. Label smoothing can also improve calibration, leading to more reliable probability estimates.

The choice of optimisation algorithms played a crucial role in training the models effectively. Initially, the **Adam optimiser** was used due to its adaptive learning rate capabilities, which can accelerate convergence, especially in the presence of noisy gradients. However, during experimentation, convergence issues and generalisation gaps were observed, with the model's validation accuracy plateauing and not improving significantly. This led to the adoption of **Stochastic Gradient Descent (SGD)** with a momentum of **0.9**. SGD with momentum has been shown to provide better generalisation performance in deep learning tasks by maintaining a consistent update direction and reducing oscillations. Additionally, SGD allows for finer control over the learning rate, which, when coupled with a learning rate scheduler, can lead to more stable convergence and improved performance on the validation set.

Learning rate scheduling was implemented using the **ReduceLROnPlateau** scheduler, which reduces the learning rate by a factor (e.g., 0.5) when the validation loss plateaus for a specified number of epochs (e.g., 3 epochs). This aids in escaping local minima and allows the model to continue learning at a finer scale, effectively fine-tuning the weights. **Early stopping** was also incorporated based on validation accuracy to prevent overfitting and reduce unnecessary computations by halting training when performance ceased to improve over a set patience period. These strategies collectively enhanced the model's convergence behaviour and generalisation capability.

### Influence of EDA on Processing

The findings from the **Exploratory Data Analysis (EDA)** directly influenced the data pre-processing and modelling approach. The high intra-class variability and inter-class similarities identified during EDA necessitated the application of robust data augmentation techniques to improve generalisation and reduce overfitting by simulating real-world variations. The EDA revealed that colour features alone were insufficient for class discrimination due to overlapping colour distributions, which influenced the decision to focus on texture and shape features, best captured by CNNs. The inability of simple models, like MLPs, to capture subtle differences further justified the adoption of pre-trained CNNs capable of learning complex, hierarchical features. Moreover, the significant variability in lighting conditions and image quality highlighted by the EDA underscored the importance of normalisation to standardise the data and careful data preparation to enhance model robustness. The blurriness threshold that I used to eliminate blurry photos from the data in pre-processing was also decided upon my the blurriness distributions I generated. This comprehensive approach ensured that the model design was closely aligned with the data characteristics revealed by the EDA.

### Evaluation of Processing Improvements

The impact of the processing improvements was evident in the model's performance metrics and convergence behaviour. **Data augmentation** and the inclusion of additional data from the Fruits 360 Dataset significantly enhanced the model's ability to generalise, as evidenced by an increase in validation accuracy and a reduction in overfitting. Transitioning to **fine-tuned pre-trained models**, particularly ResNet-50, resulted in a substantial performance boost compared to custom CNNs and the baseline MLP, with validation accuracy increasing from approximately **34% to over 94%**. Adjustments to the optimiser—from Adam to SGD with momentum—and the implementation of learning rate scheduling improved convergence rates and training stability. The learning rate scheduler allowed the model to escape local minima and continue improving, while early stopping prevented overfitting by halting training when the validation loss ceased to decrease. These processing improvements collectively contributed to a more robust and accurate classification model.

The model's validation accuracy increased from approximately 34.23% with the baseline MLP to 94.31% with the fine-tuned ResNet-50. Loss curves showed a consistent decrease in both training and validation loss after the enhancements, indicating effective learning and reduced overfitting.

These improvements underscore the efficacy of the comprehensive data pre-processing, feature design, and selection methods. The use of advanced CNN architectures, combined with fine-tuning and proper regularisation, enabled the model to effectively learn the complex features necessary for distinguishing between tomato, cherry, and strawberry images.

## Methodology

The methodology adopted for developing the image classification model encompasses meticulous data preparation, strategic network architecture design, appropriate loss functions, optimisation strategies, regularisation techniques, activation functions, hyperparameter tuning, and the utilisation of pre-trained models.

### Data Usage

The dataset was partitioned into **training and validation sets using an 80:20 split**, ensuring a robust evaluation while maintaining sufficient data for model training. This split was chosen to provide the model with ample data to learn from while preserving a representative subset for validation. The slight class imbalance introduced by the additional images was addressed using a **WeightedRandomSampler** during training. This sampler adjusts the sampling probabilities to ensure that each class is equally represented in each batch, effectively mitigating any bias and preventing the model from becoming skewed toward overrepresented classes. This approach is crucial for maintaining model fairness and ensuring balanced learning across all classes.

Data augmentation techniques were systematically applied to the training images to simulate real-world variations and prevent overfitting, as informed by the EDA findings. The transformations included:

- **Random Resized Cropping**: Simulated different scales and perspectives, helping the model become invariant to object size variations.
- **Random Horizontal Flipping**: Addressed orientation differences, ensuring the model does not become biased toward a particular direction.
- **Random Rotation up to 10 Degrees**: Simulated slight rotational variations, reflecting real-world image capture scenarios.
- **Colour Jittering**: Adjusted brightness, contrast, saturation, and hue to mimic varying lighting conditions identified during EDA.
- **Normalisation**: Applied using **ImageNet mean and standard deviation values** to standardise the input data, facilitating effective learning by ensuring consistent data distribution.

These augmentations aimed to improve the model's robustness to variations in image scale, orientation, and colour, thereby enhancing its ability to generalise to unseen data and reducing overfitting by increasing training data diversity.

### Network Architecture and Activation Functions

The final model utilized a pre-trained **ResNet-50** architecture due to its depth, residual connections, and proven performance on image classification tasks, particularly with complex datasets. **ResNet-50** is known for addressing the vanishing gradient problem through residual blocks, allowing for the training of very deep networks. The network architecture was modified as follows:

- **Layer Freezing**: The initial layers (up to `layer2`) were frozen to retain the pre-trained weights learned from the ImageNet dataset, preserving low-level feature detectors such as edges and textures that are generally applicable across different image datasets.
- **Fine-Tuning Deeper Layers**: Deeper layers (`layer3` and `layer4`) were unfrozen to allow fine-tuning. This enables the model to adjust higher-level feature representations to the specific characteristics of the tomato, cherry, and strawberry images, capturing nuanced differences identified during EDA.
- **Replacement of Final Fully Connected Layer**: The original fully connected layer was replaced with a new one comprising **three output neurons** corresponding to the three classes, with a **Softmax** activation function to output class probabilities.
- **Activation Functions**: The **Rectified Linear Unit (ReLU)** activation function was used throughout the network for its computational efficiency and ability to mitigate the vanishing gradient problem, promoting sparse activations and improving convergence.



### Loss Function

The primary loss function employed was the **Cross-Entropy Loss**, which is suitable for multi-class classification tasks as it quantifies the difference between the predicted probability distribution and the true distribution. To address potential overconfidence in predictions and improve generalization, **Label Smoothing** was incorporated with a smoothing parameter of **0.1**. Label smoothing distributes a portion of the probability mass from the true class to the other classes, assigning a small probability to incorrect classes. This technique helps prevent the model from becoming overly confident in its predictions, which can lead to overfitting, and encourages the model to be more adaptable, improving its ability to generalize to unseen data.

### Optimisation Methods and Hyperparameter Settings

The optimization strategy evolved through systematic experimentation and analysis of training dynamics:

- **Optimizer Selection**:
    - **Initial Attempt with Adam**: The **Adam optimizer** was initially utilized for its adaptive learning rates and fast convergence properties. However, it led to convergence issues and inconsistent validation performance, potentially due to its sensitivity to hyperparameter settings and sometimes poorer generalization.
    - **Switch to SGD with Momentum**: **Stochastic Gradient Descent (SGD)** with a momentum of **0.9** was adopted. SGD is known for better generalization and more stable convergence in deep learning tasks. The momentum term helps accelerate gradients in the right direction, smoothing out the oscillations.
- **Learning Rate Scheduling**:
    - **ReduceLROnPlateau Scheduler**: Implemented to dynamically adjust the learning rate. If the validation loss did not improve for **three consecutive epochs**, the learning rate was reduced by a factor of **0.5**. This approach allows for larger learning rates during the initial training phases for rapid learning and smaller rates later to fine-tune the weights.
- **Hyperparameter Settings**:
    - **Initial Learning Rate**: Set to **0.001** for the unfrozen layers and **0.0001** for the pre-trained layers, allowing for cautious updates to the pre-trained weights.
    - **Batch Size**: Chosen as **32** to balance memory constraints and training stability.
    - **Number of Epochs**: Training conducted for up to **50 epochs** with early stopping based on validation accuracy.
    - **Weight Decay**: L2 regularization applied with a weight decay parameter of **5e-4** to prevent overfitting.
    - **Label Smoothing**: Applied with a parameter of **0.1**.

These optimization methods and hyperparameter settings were carefully tuned to achieve optimal performance and stable convergence.

Hyperparameters were carefully tuned to optimise performance:

- **Initial Learning Rate**: Set to 0.001, determined through a learning rate range test.
- **Batch Size**: Set to 32, balancing training stability and computational efficiency.
- **Number of Epochs**: Training conducted for up to 50 epochs with early stopping based on validation accuracy.
- **Weight Decay**: L2 regularisation applied with a weight decay parameter of 5e-4.
- **Layer-Specific Learning Rates**: Lower learning rate applied to pre-trained layers (1e-4) and higher learning rate to the new fully connected layer (1e-3).

*Table 1: Hyperparameter Settings*

| Hyperparameter          | Value      |
|-------------------------|------------|
| Initial Learning Rate   | 0.001      |
| Batch Size              | 32         |
| Number of Epochs        | Up to 50   |
| Dropout Rate            | 0.3        |
| Weight Decay            | 5e-4       |
| Momentum (SGD)          | 0.9        |
| Label Smoothing         | 0.1        |

### Regularisation Strategies

Several regularisation techniques were employed to prevent overfitting and improve model generalisation:

- **Dropout**: A dropout rate of **0.3** was applied after the fully connected layers in custom CNN models. Dropout randomly sets a fraction of input units to zero during training, which prevents units from co-adapting too much and encourages the network to learn more robust features.
- **Data Augmentation**: As previously detailed, data augmentation significantly increased the diversity of the training data, simulating various real-world conditions and reducing overfitting by exposing the model to a broader set of scenarios.
- **Early Stopping**: Monitored the validation accuracy with a patience of **5 epochs** to cease training if no improvement was observed. This prevents the model from overfitting to the training data by stopping training before the performance on validation data degrades.
- **L2 Regularisation (Weight Decay)**: Applied through a weight decay parameter of **5e-4** in the optimiser. L2 regularisation adds a penalty proportional to the square of the magnitude of the weights, discouraging complex models with large weights and thus reducing overfitting.
- **Label Smoothing**: As previously discussed, helped prevent overconfidence in predictions.

These regularisation strategies collectively contributed to improved model generalisation and robustness.

### Use of Existing Models

**Transfer Learning and Pre-Trained Models**:

- **Model Used**: Pre-trained **ResNet-50** trained on the **ImageNet** dataset.
- **Fine-Tuning Strategy**:
    - **Layer Freezing and Unfreezing**: Initially, earlier layers were frozen to retain general feature representations. Deeper layers (`layer3` and `layer4`) were unfrozen to fine-tune and adapt the model to the specific features of the tomato, cherry, and strawberry images.
    - **Replacement of Final Layer**: The original classification layer was replaced with a new fully connected layer with **three output neurons** corresponding to the target classes.
- **Advantages**:
    - **Leveraged Learned Features**: Benefited from the rich feature representations learned from ImageNet, which includes a wide variety of images, accelerating learning and improving performance.
    - **Reduced Training Time**: Required less data and computational resources compared to training a deep network from scratch.
    - **Improved Generalisation**: Pre-trained models often generalise better due to exposure to diverse datasets.

This approach effectively utilised existing models to enhance performance while adapting to the specific requirements of the classification task.

### Performance Evaluation

The training and validation accuracy over epochs, presented in **Figures 5 and 6**, illustrate the model's convergence behaviour and the effectiveness of early stopping. The plots show that the training accuracy steadily increases while the validation accuracy improves and then plateaus, indicating that the model is learning effectively without overfitting. Early stopping was triggered when the validation accuracy did not improve for **3 consecutive epochs**, preventing unnecessary training and potential overfitting.

*Figure 5 and 6: CNN Training and Validation Loss over Epochs, CNN Training and Validation Accuracy over Epochs *

![[COMP309 Project Report-20241029154207512.webp|250]]![[COMP309 Project Report-20241029154229952.webp|250]]

The **confusion matrix** (Figure 7) provides insights into class-specific performance, highlighting how well the model distinguishes between each class. The matrix shows the number of correct and incorrect predictions for each class, allowing for the identification of any classes that are more challenging to predict. For instance, if the model occasionally confuses cherries with strawberries, this would be reflected in the off-diagonal elements and could indicate a need for further fine-tuning or additional data augmentation focused on those classes.

*Figure 7: Confusion Matrix on Validation Set.*

![[COMP309 Project Report-20241029154402016.webp|250]]

**Final Model Performance**:

- **Validation Accuracy**: **94.31%**
- **Precision, Recall, F1 Score**: Precision: **0.9449**, Recall: **0.9431**, F1 Score: **0.9432**
- **Average Loss**: **0.4224**

### Justification of Choices

The choices made throughout the methodology were carefully justified based on empirical results and the specific challenges of the task:

- **Data Augmentation**: Implemented to address the high variability in the dataset identified during EDA, such as variations in lighting, orientation, and scale. Augmentation improved generalisation by exposing the model to a wider range of scenarios.
- **Use of Pre-Trained Models (Transfer Learning)**: Leveraged pre-trained **ResNet-50** to benefit from learned feature representations, accelerating convergence, and improving performance on a limited dataset.
- **Optimisation Methods**: Switched to **SGD with momentum** after observing convergence issues with Adam, leading to enhanced convergence stability and better generalisation.
- **Regularisation Techniques**: Applied dropout, weight decay, and label smoothing to prevent overfitting, as the model complexity increased with the use of deep networks.
- **Hyperparameter Tuning**: Systematic tuning of learning rates, batch sizes, and other hyperparameters optimised performance, with choices guided by validation metrics and training behaviour.
- **Fine-Tuning Strategy**: Unfreezing deeper layers allowed the model to adapt high-level features to the specific dataset, crucial for distinguishing between visually similar classes.

These decisions collectively contributed to the model's high performance and robustness.

## Summary and Discussion

The project involved a comparative analysis between the baseline **Multilayer Perceptron (MLP)** model and the best-performing **Convolutional Neural Network (CNN)** model, highlighting differences in architectures, training times, and classification performances. This comparison underscores the impact of model selection on the ability to classify visually similar images effectively.

**Baseline MLP Structure and Settings**:

- **Architecture**:
    - **Input Layer**: Images resized to **64×64 pixels** and flattened into a vector of size 12,288.
    - **Hidden Layers**:
        - **First Hidden Layer**: 512 neurons with **ReLU** activation function to introduce non-linearity.
        - **Second Hidden Layer**: 256 neurons with **ReLU** activation.
    - **Output Layer**: 3 neurons corresponding to the three classes, with a **Softmax** activation function to output class probabilities.
- **Regularisation**:
    - **Dropout**: Applied with a rate of **0.5** after each hidden layer to prevent overfitting by randomly deactivating neurons during training.
- **Loss Function**: **Cross-Entropy Loss**, suitable for multi-class classification.
- **Optimiser**: **Adam optimiser** with a learning rate of **0.001** for its adaptive learning rate properties.
- **Training Parameters**:
    - **Batch Size**: **32** to balance computational efficiency and training stability.
    - **Number of Epochs**: Trained for up to **50 epochs** with early stopping based on validation loss to prevent overfitting.

**Best CNN Model Description**:

- **Architecture**: Fine-tuned **ResNet-50** pre-trained on ImageNet.
    - **Layer Freezing**:
        - **Frozen Layers**: All layers up to and including `layer2` to retain general low-level features.
    - **Fine-Tuning**:
        - **Unfrozen Layers**: `layer3` and `layer4` to adapt high-level features to the specific dataset.
    - **Modified Fully Connected Layer**:
        - Replaced the original final layer with a new fully connected layer with **3 output neurons** corresponding to the classes, using a **Softmax** activation function.
- **Activation Functions**:
    - **ReLU** used throughout the network for non-linearity and to address the vanishing gradient problem.
- **Loss Function**:
    - **Cross-Entropy Loss** with **label smoothing** (parameter 0.1) to improve generalisation.
- **Optimiser**:
    - **SGD with momentum** (momentum=0.9) for better convergence and generalisation.
- **Training Parameters**:
    - **Batch Size**: **32**.
    - **Number of Epochs**: Up to **50 epochs** with early stopping based on validation accuracy.

### Comparison of MLP and CNN Models

**Training Time**:

- **MLP**: Trained faster due to its simpler architecture and fewer parameters, with each epoch taking approximately **10 seconds** on the available hardware.
- **CNN (ResNet-50)**: Required more computational resources and time per epoch due to its depth and complexity, with each epoch taking around **80 seconds**. The increased training time is justified by the significant improvement in performance.

**Classification Performance**:

- **MLP**:
    - **Validation Accuracy**: Achieved a maximum of **34.23%**, indicating that the model struggled to learn complex patterns necessary for distinguishing between visually similar classes.
    - **Limitations**: The MLP could not capture spatial hierarchies and local patterns due to the flattening of image data, leading to loss of spatial information.
- **CNN (ResNet-50)**:
    - **Validation Accuracy**: Achieved **94.31%**, demonstrating effective learning and good generalisation to unseen data.
    - **Advantages**: The CNN's convolutional layers effectively captured spatial hierarchies and complex features, crucial for the task.

**Classification Performance**: The MLP achieved a maximum validation accuracy of 34.23%, struggling to learn complex patterns, and extremely overfitting to the training data. The CNN achieved a validation accuracy of 94.31%, displaying effective learning and good generalisation.

*Figure 8: MLP Training and Validation Loss over Epochs.*
*Figure 9: MLP Training and Validation Accuracy over Epochs.*

![[COMP309 Project Report-20241028131110501.webp|250]]![[COMP309 Project Report-20241028131110611.webp|250]]


**Striking Differences Between MLP and CNN Models**:

- **Model Complexity and Architecture**:
    - **MLP**: Simpler architecture, treats all input pixels equally without considering spatial relationships.
    - **CNN**: Complex architecture with convolutional layers that capture spatial hierarchies and local patterns.
- **Feature Extraction Capability**:
    - **MLP**: Lacks the ability to automatically extract and learn features from raw image data.
    - **CNN**: Automatically learns hierarchical feature representations, from edges and textures to complex shapes.
- **Generalisation to Unseen Data**:
    - **MLP**: Under-fit the data, resulting in poor performance on validation data.
    - **CNN**: Demonstrated strong generalisation, accurately classifying unseen images.
- **Training Efficiency and Resource Requirements**:
    - **MLP**: Faster training times but at the cost of lower accuracy.
    - **CNN**: Longer training times and higher computational demands, justified by significantly higher accuracy.

### Discussion of Differences

The CNN's superior performance is attributed to several key factors:

- **Effective Capture of Spatial Hierarchies**: The convolutional and pooling layers in CNNs preserve spatial relationships, enabling the model to recognise patterns and features crucial for distinguishing between visually similar classes.
- **Transfer Learning**: Leveraged features learned from the large and diverse **ImageNet** dataset, providing a strong foundation for feature extraction that was fine-tuned for the specific task.
- **Robust Regularisation Techniques**: Use of dropout, weight decay, and label smoothing prevented overfitting despite the model's complexity.
- **Advanced Optimisation Techniques**: Adoption of **SGD with momentum** and learning rate scheduling enhanced convergence and stability, leading to better generalisation.

In contrast, the MLP lacked the capability to exploit spatial information due to the flattening of image data, resulting in the loss of spatial hierarchies. This limitation caused the MLP to under-fit the data, as it could not learn the complex patterns necessary for accurate classification, leading to poor performance.

## Conclusions and Future Work

**Summary of Conclusions**:

- **CNNs Significantly Outperform MLPs**: In image classification tasks involving visually similar classes, CNNs' ability to capture spatial hierarchies and complex features results in superior performance.
- **Effectiveness of Transfer Learning**: Fine-tuning pre-trained models like **ResNet-50** is highly effective, leveraging learned features to achieve high accuracy even with limited dataset sizes.
- **Necessity of Advanced Techniques**: Simple models and basic feature extraction methods are insufficient for complex image data. Advanced modelling techniques, careful pre-processing, and thorough hyperparameter tuning are essential to achieve optimal performance.
- **Importance of Data Augmentation and Regularization**: Implementing robust data augmentation and regularization strategies is crucial to prevent overfitting and enhance generalization.

### Pros and Cons of the Approach

**Pros**:

- **High Performance**: The model achieved a validation accuracy of **94.31%**, demonstrating effective classification of visually similar classes.
- **Leveraging Pre-Trained Models**: Utilizing pre-trained models reduced training time and computational resources compared to training from scratch while enhancing performance.
- **Improved Generalization and Robustness**: Data augmentation and regularization techniques effectively prevented overfitting and improved the model's ability to generalize to unseen data.
- **Adaptability**: Fine-tuning allowed the model to adapt pre-trained features to the specific dataset.

**Cons**:

- **Increased Computational Demand**: The use of deep CNNs requires more computational resources, both in terms of memory and processing power, which may not be available in all environments.
- **Complexity in Hyperparameter Tuning**: The increased number of hyperparameters in deep networks makes the tuning process more complex and time-consuming.
- **Dependence on Pre-Trained Models**: Reliance on models trained on datasets like ImageNet may introduce biases or limit the model's ability to capture dataset-specific nuances that are not represented in the pre-trained data.

### Future Work

**Possible considerations for future work**:

- **Explore Alternative Architectures**:
    - **EfficientNet**: Investigate EfficientNet architectures for better parameter efficiency and potential improvements in accuracy.
    - **DenseNet and MobileNet**: Consider these architectures for their efficiency and suitability for deployment on devices with limited resources.
- **Advanced Hyperparameter Optimization**:
    - Utilize **Grid Search**, **Random Search**, or **Bayesian Optimization** to systematically explore the hyperparameter space and identify optimal settings.
- **Implement Ensemble Methods**:
    - Combine predictions from multiple models to potentially enhance performance and robustness.
- **Expand the Dataset**:
    - Utilize **Generative Adversarial Networks (GANs)** to generate synthetic images, augmenting the dataset and introducing more variability.
- **Experiment with Advanced Regularization Techniques**:
    - Explore techniques like **DropBlock**, **Cutout**, or **Mixup** to prevent overfitting.
- **Improve Model Interpretability**:
    - Use tools like **Grad-CAM** or **SHAP values** to understand which features the model relies on, enhancing interpretability and trust.
- **Optimise for Deployment**:
    - Apply model compression techniques such as **quantisation** or **pruning** to reduce model size and improve inference time, facilitating deployment on edge devices.

By pursuing these future work avenues, the model can be further refined, made more efficient, and adapted for practical, real-world applications.