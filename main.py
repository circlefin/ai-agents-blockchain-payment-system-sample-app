import subprocess
import json
import re
from ai_research import run_research

def run_circle_payment(contributions):
    process = subprocess.Popen(['node', 'circle_payment.js'], 
                               stdin=subprocess.PIPE, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=json.dumps(contributions).encode())
    
    print("Raw stdout:", stdout.decode())
    print("Raw stderr:", stderr.decode())
    
    if process.returncode != 0:
        print(f"Error from payment service: {stderr.decode()}")
        return None
    
    try:
        # Extract JSON output using regex
        json_output = re.search(r'=== JSON_OUTPUT_START ===\n(.*)\n=== JSON_OUTPUT_END ===', stdout.decode(), re.DOTALL)
        if json_output:
            results = json.loads(json_output.group(1))
            if results['success']:
                print("All payments processed successfully.")
                for agent, payment in results['payments'].items():
                    print(f"Agent: {agent}")
                    print(f"  Transaction ID: {payment['transactionId']}")
                    print(f"  Amount: {payment['amount']}")
                    print(f"  Validated: {'Yes' if payment['validated'] else 'No'}")
            else:
                print(f"Payment processing failed: {results.get('error', 'Unknown error')}")
            return results
        else:
            print("Could not find JSON output in the response")
            return None
    except json.JSONDecodeError:
        print("Failed to parse JSON output from payment service")
        return None

if __name__ == "__main__":
    topic = "Artificial intelligence and employment trends"
    contributions = run_research(topic)
    
    print("Research completed. Processing payments...")
    payment_result = run_circle_payment(contributions)
    
    if payment_result and payment_result['success']:
        print("Payment process completed successfully.")
    else:
        print("Payment process failed.")