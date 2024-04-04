import requests
from datetime import datetime, timedelta

def get_clone_counts(repo_owner, repo_name, since_date):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/traffic/clones"
    params = {'per': 'day', 'since': since_date}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch clone data. Status code: {response.status_code}")
        return None

def calculate_accumulative_sum(clone_data):
    accumulative_sum = 0
    for day_data in clone_data['clones']:
        accumulative_sum += day_data['count']
    return accumulative_sum

if __name__ == "__main__":
    repo_owner = "sinamoghimi73"
    repo_name = "GazeGenesis"
    since_date = (datetime.utcnow() - timedelta(days=365)).isoformat()  # Change the number of days as needed
    clone_data = get_clone_counts(repo_owner, repo_name, since_date)
    if clone_data:
        accumulative_sum = calculate_accumulative_sum(clone_data)
        print(f"Accumulative sum of clones: {accumulative_sum}")
