# ds-thesis-project
A Github repository for Yiyan Hao's Data Science Honors Thesis Project at UC Berkeley (Fa23-Sp24)

In this work, we evaluated the performance of general and fine-tuned large language model (LLM)-based tools for automated extraction of pathology outcomes from breast pathology reports. Our fine-tuned Mistral-7B obtained high accuracy across all tasks, which demonstrated its potential for large-scale automation of the labeling process, simplifying the labor-intensive annotation tasks in deep learning research for advancing precision health. 

The datasets used for this study are protected health information (PHI) owned and managed by the University of California, San Francisco (UCSF). Upon institutional review board (IRB) approval, the author is granted access to and authorized to work with the data through the Wynton PHI HPC environment. These data are not available for public access.

This repository documents the scripts used to preprocess the breast pathology data, evaluate LLMs, and fine-tune with hyperparameter search, under /scripts.

To run the code, first install pytorch with pip or conda. Then install unsloth (https://github.com/unslothai/unsloth.git). Then install all dependencies in this library.