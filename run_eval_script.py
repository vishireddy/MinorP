import sys
import os
from dotenv import load_dotenv

load_dotenv(override=True)
from src.evaluate import run_evaluation_suite

print("Starting Evaluation Suite automatically...")
run_evaluation_suite()
print("Evaluation Complete! Results saved to data/eval_results_v2.json")
