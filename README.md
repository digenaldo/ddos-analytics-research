# DDoS Attack Analytics Research Project

## Overview
The threat of application layer DDoS attacks is escalating, underscoring the need for effective detection methods to safeguard networked systems. With attackers constantly evolving their techniques, cybersecurity efforts are becoming more challenging. In response, leveraging advanced artificial intelligence technologies emerges as a promising avenue for bolstering defenses.

## Project Description
This project presents a detection scheme utilizing machine learning and assesses its performance using big data analytics. Four classification algorithms are evaluated for their accuracy and execution time in detecting DDoS attacks:

- Naive Bayes
- Decision Tree
- Logistic Regression
- Random Forest

Preliminary findings suggest the system's efficacy, showcasing high accuracy levels while maintaining reasonable response times. This integrated approach, marrying sophisticated data analysis with big data capabilities, shows potential in fortifying defenses against cyber threats within an increasingly intricate and dynamic landscape.

## Files in the Repository
- **abd**
  - **data_processing.py**: Contains data processing functions.
  - **evaluation.py**: Provides evaluation metrics and functions.
  - **main.py**: Main script for running the detection scheme using Spark.
  - **model_training.py**: Includes functions for training machine learning models.
  - **spark_setup.py**: Configures the Spark environment.
- **apg**
  - **analysis.py**: Script for additional analytics and visualization.
  - **evaluation.py**: Contains evaluation metrics and functions specific to the APG module.
  - **main.py**: Main script for running additional analytics and performance evaluation.
  - **models.py**: Includes machine learning models for the APG module.
  - **preprocessing.py**: Provides preprocessing functions for the APG module.
- **data**
  - **train_mosaic.csv.zip**: Training dataset.
  - **test_mosaic.csv.zip**: Test dataset.

## Usage
To run the scripts, follow the steps below:

1. Ensure you have Python installed on your system.
2. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```
3. Navigate to the `src` directory and run the Python script:
   ```bash
   cd src
   python Main.py
   ```

## Results
The results from the analysis include the accuracy and execution times of the four classification algorithms. The findings indicate that the integrated approach is effective in detecting DDoS attacks with high accuracy and reasonable response times.

## Conclusion
This project demonstrates the potential of using machine learning and big data analytics to enhance cybersecurity defenses against application layer DDoS attacks. The promising results encourage further research and development in this domain to stay ahead of evolving cyber threats.

## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries or contributions, please contact [digenaldo.rangel@gmail.com].