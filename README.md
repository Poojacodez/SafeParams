Project Description

This project demonstrates potential vulnerabilities in a machine learning (ML) service, specifically focusing on how an attacker could craft inputs to manipulate the service's predictions. It simulates a simplified ML service and shows how it can be tricked. This is an input validation project.

Key Components and Functionality

Sample Data Generation:

The generate_sample_data function creates synthetic data for training and testing.

It generates a set of input features (X) and corresponding binary labels (y) based on a simple rule.

This function uses numpy for numerical operations and random number generation.

Vulnerable ML Service:

The VulnerableMLService class simulates an ML service with a trained model.

It uses a RandomForestClassifier from scikit-learn as the underlying model.

The service includes a predict method that takes input data and returns predictions.

Vulnerability: The service has an optional input validation mechanism. When enabled, it checks if the input features fall within the range of the training data. However, this validation can be bypassed.

The service calculates and prints the accuracy of the trained model on a test set.

Attack Demonstration:

The if __name__ == "__main__": block demonstrates how an attacker could exploit the vulnerabilities.

It creates an instance of the VulnerableMLService.

It shows several attack scenarios:

Legitimate Input: Shows the service working correctly with valid input.

Adversarial Input (with validation): Shows how the service detects and rejects an input with extreme values when validation is enabled.

Adversarial Input (without validation): Shows how an attacker can bypass the validation and cause the service to make an incorrect prediction.

Stealthy Adversarial Input: Demonstrates a more subtle attack where the input is crafted to be within the expected range (bypassing validation) but still causes a targeted incorrect prediction.

Type Confusion Attack: Shows what happens if the service receives an input of the wrong data type (e.g., a string instead of a numerical array). This is a common vulnerability.

Purpose

The primary purpose of this project is to illustrate the importance of robust input validation and security considerations in machine learning systems. It highlights how failing to properly validate inputs can leave ML services vulnerable to various attacks. It's a simplified example to make the concepts clear.

Libraries Used

numpy: For numerical operations (array manipulation, etc.).

pandas: (Not directly used in the provided code, but commonly used with scikit-learn).

scikit-learn: For the machine learning model (RandomForestClassifier) and splitting data.
