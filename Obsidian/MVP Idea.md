### **Project Summary**  

#### **Initial Focus**

- Use the **EMBER dataset** for feature-based malware poisoning due to its ease of use and feature extraction tools.
- Implement the project in Python, leveraging Docker for portability.

#### **Planned Steps**  

1. **Baseline Experiment**  
   - Train a benign static malware classification model using features from the EMBER dataset.  
   - Test poisoning pipeline with simple attacks, such as:
     - **Label-flipping**: Randomly mislabel benign/malicious samples.
     - **Distance-based random flipping**: Modify samples based on feature proximity.  

2. **Feature Modification**  
   - Modify **features** directly in the EMBER dataset, skipping binary/source code manipulation for now.  
   - Explore **evolutionary programming** to generate perturbations:
     - Define mutation operations based on feature types (e.g., headers, imports, size).
     - Optimise perturbations to add imperceptible noise that maintains malicious functionality.  

3. **Future Directions**  
   - **Binaries and Cuckoo Sandbox**:
     - Use binaries from **TheZoo** to manipulate actual malware files.
     - Validate functionality using **Cuckoo Sandbox** (e.g., behavioural checks).  
   - **Imperceptibility**:  
     - Develop metrics to ensure modifications are hard to detect while maintaining model effectiveness.

4. **Challenges**  
   - **Functionality**: Ensuring malware works post-modification is complex but deferred for now.  
   - **Imperceptibility Testing**: Measure how perturbations impact the model and remain undetectable.
