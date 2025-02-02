import os
from azure.identity import DefaultAzureCredential
from azure.ai.evaluation import evaluate, GroundednessEvaluator, RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator, SimilarityEvaluator, F1ScoreEvaluator
from custom_eval._helpfulness import HelpfulnessEvaluator
from response import generate_answer
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import json

credential = DefaultAzureCredential()
# Umgebungsvariablen
def environmentVariables():
    load_dotenv()
    azure_ai_project = {
        "subscription_id": os.environ.get("AZURE_SUBSCRIPTION_ID"),
        "resource_group_name": os.environ.get("AZURE_RESOURCE_GROUP"),
        "project_name": os.environ.get("AZURE_PROJECT_NAME"),
    }
    model_config = {
        "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
    }
    return azure_ai_project, model_config
"""
Groundedness:   Misst, wie gut die generierte Antwort mit dem gegebenen Kontext in einem retrieval-augmented Generation-Szenario übereinstimmt, wobei der Schwerpunkt auf Relevanz und Genauigkeit in Bezug auf den Kontext liegt.
Relevance:      Misst, wie effektiv eine Antwort auf eine Anfrage eingeht. Sie bewertet die Genauigkeit, Vollständigkeit und direkte Relevanz der Antwort ausschließlich basierend auf der gegebenen Anfrage.
Coherence:      Misst die logische und geordnete Darstellung von Ideen in einer Antwort, sodass der Leser dem Gedankengang des Schreibers leicht folgen und ihn verstehen kann.
Fluency:        Misst die Effektivität und Klarheit der schriftlichen Kommunikation, mit Fokus auf grammatikalische Genauigkeit, Wortschatzumfang, Satzkomplexität, Kohärenz und allgemeine Lesbarkeit.
Similarity:     Misst den Grad der Ähnlichkeit zwischen dem generierten Text und dem Ground Truth in Bezug auf eine Anfrage.
F1 Score:       Misst die Ähnlichkeit durch gemeinsam genutzte Token zwischen dem generierten Text und dem Ground Truth, wobei sowohl Präzision als auch Recall im Fokus stehen.
Helpfulness:    Misst, wie gut die generierte Antwort dem Nutzer bei der Beantwortung seiner Anfrage hilft.
"""
def initialize_evaluators(model_config):
    groundedness_eval = GroundednessEvaluator(model_config)
    relevance_eval = RelevanceEvaluator(model_config)
    coherence_eval = CoherenceEvaluator(model_config)
    fluency_eval = FluencyEvaluator(model_config)
    similarity_eval = SimilarityEvaluator(model_config)
    f1score_eval = F1ScoreEvaluator()
    helpfulness_eval = HelpfulnessEvaluator(model_config)
    return {
        "groundedness": groundedness_eval,
        "relevance": relevance_eval,
        "coherence": coherence_eval,
        "fluency": fluency_eval,
        "similarity": similarity_eval,
        "f1score": f1score_eval,
        "helpfulness": helpfulness_eval
    }

# Definition: CSV zu JSONL
def csv_to_jsonl(csv_file_path, jsonl_file_path):
    df = pd.read_csv(csv_file_path)
    df = df.fillna('')
    if 'response' in df.columns:
        df = df.drop(columns=['response'])
    with open(jsonl_file_path, 'w') as jsonl_file:
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            json_str = json.dumps(row_dict)
            jsonl_file.write(json_str + '\n')

if __name__ == '__main__':
    azure_ai_project, model_config = environmentVariables()
    evaluators = initialize_evaluators(model_config)
    
    # Aktion: CSV zu JSONL
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, 'dataset', 'dataset_mb.csv')
    jsonl_file_path = os.path.join(current_dir, 'dataset', 'dataset_mb.jsonl')
    csv_to_jsonl(csv_file_path, jsonl_file_path)

    models = ["gpt-4o", "gpt-4"] # Modelle, die verglichen werden --> Kann beliebig erweitert werden
    for model in models:
        evaluation_name = f"{model}_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        """
        Query: Die an die generative KI-Anwendung gesendete Anfrage
        Response: Die von der generativen KI-Anwendung auf die Anfrage generierte Antwort
        Context: Die Quelle, auf der die generierte Antwort basiert (das heißt, die zugrunde liegenden Dokumente)
        Ground truth: Die vom Nutzer/Menschen erstellte Antwort als wahre Lösung
        Conversation: Eine Liste von Nachrichten mit abwechselnden Beiträgen von Nutzer und Assistent. Siehe mehr im nächsten Abschnitt.
        """
        result = evaluate(
            evaluation_name=evaluation_name,
            data=jsonl_file_path, # provide your data here
            target=generate_answer(**model_config, model=model),
            evaluators=evaluators,
            # column mapping
            evaluator_config={
                "groundedness": {
                    "column_mapping": {
                        "query": "${data.query}",
                        "context": "${data.user_instructions}"+"\n"+"${data.context}",
                        "response": "${target.response}"
                    } 
                },
                "relevance": {
                    "column_mapping": {
                        "query": "${data.query}",
                        "response": "${target.response}"
                    } 
                },
                "coherence": {
                    "column_mapping": {
                        "query": "${data.query}",
                        "response": "${target.response}"
                    } 
                },
                "fluency": {
                    "column_mapping": {
                        "response": "${target.response}"
                    } 
                },
                "similarity": {
                    "column_mapping": {
                        "query": "${data.query}",
                        "response": "${target.response}",
                        "ground_truth": "${data.ground_truth}"
                    } 
                },
                "f1score": {
                    "column_mapping": {
                        "response": "${target.response}",
                        "ground_truth": "${data.ground_truth}"
                    } 
                },
                "helpfulness": {
                    "column_mapping": {
                        "query": "${data.query}",
                        "context": "${data.user_instructions}"+"\n"+"${data.context}",
                        "response": "${target.response}"
                    } 
                }
            },
            azure_ai_project=azure_ai_project,
            output_path = os.path.join(current_dir, f"results_{evaluation_name}.json")
        )