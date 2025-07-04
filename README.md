
# ML Input Validation Vulnerability Simulation

This project demonstrates potential vulnerabilities in a machine learning (ML) service, focusing on how attackers could craft inputs to manipulate the service's predictions. It simulates a simplified ML service and illustrates various techniques to bypass weak input validation mechanisms.

---

## Key Components and Functionality

### Sample Data Generation

* The `generate_sample_data` function creates synthetic training and testing data.
* It generates:

  * A set of input features (`X`)
  * Corresponding binary labels (`y`) based on a simple rule.
* Utilizes **NumPy** for array manipulation and random number generation.

### Vulnerable ML Service

* The `VulnerableMLService` class simulates an ML prediction API using:

  * `RandomForestClassifier` from **scikit-learn** as the underlying model.
* Includes a `predict` method for making predictions on input data.
* Input validation mechanism:

  * When enabled, the service checks if input features fall within the range of the training data.
  * This validation is optional and can be bypassed.
* The model calculates and prints its accuracy on a test dataset.

### Attack Demonstration

The main function (`if __name__ == "__main__":`) demonstrates multiple attack scenarios:

1. **Legitimate Input**
   The service processes and responds correctly to valid data.

2. **Adversarial Input (with validation enabled)**
   The service correctly detects and blocks inputs with extreme values.

3. **Adversarial Input (with validation disabled)**
   Attackers can bypass validation, resulting in incorrect predictions.

4. **Stealthy Adversarial Input**
   Crafted inputs stay within acceptable ranges but still cause targeted misclassification.

5. **Type Confusion Attack**
   Demonstrates how incorrect input types (e.g., strings instead of arrays) can lead to unexpected behavior or failure.

---

## Purpose

The goal of this project is to highlight the importance of robust input validation in machine learning systems. It demonstrates how weak or improperly configured validation mechanisms can leave ML services vulnerable to various attacks. This simulation provides a simplified yet informative view of real-world security challenges in deploying ML models.

---

## Libraries Used

* **NumPy** – For numerical computations and data generation
* **scikit-learn** – For training the Random Forest model and splitting datasets

---

