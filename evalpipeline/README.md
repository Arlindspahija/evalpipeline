AzureLLMEvaluator
AzureLLMEvaluator is a project designed to simplify the evaluation of Large Language Models (LLMs) deployed on Azure AI Foundry. It provides an easy-to-use interface or framework for analyzing, testing, and benchmarking LLM performance across various scenarios, ensuring you get the most out of your Azure-hosted AI models.

Introduction
To thoroughly assess the performance of your generative AI application, especially when applied to a substantial dataset, AzureLLMEvaluator helps you evaluate your Generative AI application in your development environment using the Azure AI evaluation SDK. Given either a test dataset or a target, the generations of your AI application are quantitatively measured with both mathematical-based metrics and AI-assisted quality and safety evaluators. Built-in or custom evaluators provide you with comprehensive insights into the application's capabilities and limitations.

In this project, you will learn how to run evaluators on a single row of data, apply a larger test dataset to an application target with built-in evaluators, and use the Azure AI evaluation SDK both locally and remotely in the cloud. Furthermore, you can track the results and evaluation logs in an Azure AI project for detailed performance analysis.

For more information on how to use the Azure AI evaluation SDK, check out the official documentation.

Features
Comprehensive Evaluation: Measure LLM performance using metrics like groundedness, relevance, coherence, fluency, similarity, F1 score, and helpfulness.
Customizable Scenarios: Evaluate your models in different contexts with the ability to configure queries, responses, and grounding documents.
Integration with Azure: Seamlessly integrate with Azure AI services, utilizing the Azure OpenAI endpoints and models for evaluation.
CSV to JSONL Conversion: Automatically convert CSV datasets into JSONL format for easy evaluation.
Result Tracking: Optionally track evaluation results using Azure AI projects for more detailed monitoring.
Setup
Clone the repository:

git clone https://github.com/yourusername/AzureLLMEvaluator.git
cd AzureLLMEvaluator
Install dependencies:

pip install -r requirements.txt
Set up environment variables:

Create a .env file in the root directory with the following content:

AZURE_SUBSCRIPTION_ID="your-subscription-id"
AZURE_RESOURCE_GROUP="your-resource-group"
AZURE_PROJECT_NAME="your-project-name"
AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"    
Run the evaluation pipeline:

The main evaluation script (evaluation_pipeline.py) will execute the evaluation of LLM responses against the provided dataset. It will generate results that are outputted in JSON format, summarizing various performance metrics.

python evaluation_pipeline.py
Usage
The evaluation_pipeline.py script is the core of the evaluation process. You can customize it by adjusting the following parameters:

Models: Specify the models to be evaluated (e.g., gpt-4, gpt-4o).
Dataset: Provide the dataset (CSV format) for testing the models.
Evaluators: Define which evaluators to use for different quality assessments (e.g., groundedness, relevance, fluency, etc.).
Evaluators
The project includes various evaluators for assessing different aspects of response quality:

Groundedness: Measures how well the generated response aligns with the context.
Relevance: Assesses the relevance of the response to the query.
Coherence: Evaluates the logical flow and clarity of the response.
Fluency: Assesses the grammatical accuracy and readability.
Similarity: Measures the similarity between generated responses and ground truth.
F1 Score: Evaluates precision and recall.
Helpfulness: Measures how well the response addresses the user's query.
You can customize which evaluators to use for your specific needs, depending on the type of analysis you want to perform.

Result Tracking
The evaluation results will be saved in the output directory with a timestamped filename. You can track metrics and download the full evaluation report as a JSON file. This allows for easy monitoring of model performance across different evaluation runs. In the same file you will see a link to look at the results in AI Foundry, this way you can compare results of multiple evaluations.

Contributing
Contributions are welcome! If you want to improve the project or add new features, feel free to submit issues or pull requests. To contribute:

Fork the repository.
Make your changes in a feature branch.
Submit a pull request with a description of your changes.
Please ensure that your contributions follow the project's coding conventions and include tests where appropriate.