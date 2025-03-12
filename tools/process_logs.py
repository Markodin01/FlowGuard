import json

def read_logs(file_path):
    logs = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("INFO:__main__:System State: "):
                log_str = line[len("INFO:__main__:System State: "):].strip()
                log_dict = eval(log_str)
                logs.append(log_dict)
    return logs

def process_logs(logs):
    processed_logs = {}
    for log in logs:
        step = log['Step']
        valve_states = [state // 5 for state in log['Valve States']]
        alarms = log['Alarms']
        processed_entry = valve_states + alarms
        processed_logs[step] = processed_entry
    return processed_logs

def main():
    logs = read_logs('agent_and_environment.log')
    processed_logs = process_logs(logs)
    
    with open('env_states.json', 'w') as outfile:
        json.dump(processed_logs, outfile, indent=4)
    
    print("Processed logs have been saved to env_states.json")

if __name__ == "__main__":
    main()
