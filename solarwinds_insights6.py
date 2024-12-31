from flask import Flask, request, jsonify, json

app = Flask(__name__)

chatgpt_responses = [] # To store question-response pairs

#----------------------FUNCTION TO ASK CHATGPT-------------------------

import requests

def send_to_chatgpt(message):
    chatgpt_api_url = "https://api.openai.com/v1/chat/completions"
    chatgpt_api_key = "sk-"

    headers = {
        "Authorization": f"Bearer {chatgpt_api_key}",
        "Content-Type": "application/json"
    }


    data = {
        "model": "gpt-4-0125-preview",
        "messages": [{"role": "system", "content": "You are an expert in Cisco Network Engineering specializing in syslogs and SNMP traps. Please suggest two or three immediate troubleshooting actions. Keep the response to less than 250 tokens"},
    {"role": "user", "content": message}],
        "max_tokens": 330,
        "temperature": .5
    }

    print("HTTP POST Request Data to ChatGPT API:")
    print(json.dumps(data, indent=4))  # Pretty print the JSON data

    response = requests.post(chatgpt_api_url, headers=headers, json=data)

    if response.status_code == 200:
        chatgpt_response = response.json()
        return chatgpt_response["choices"][0]["message"]["content"]
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")
        return "Error: ChatGPT request failed"

#----------------------FUNCTION TO SEND TO WEBEX------------------------------

def send_to_webex(question, ip_address, node, response):
    webex_api_url = "https://webexapis.com/v1/webhooks/incoming/Y2"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "roomId": "Solarwinds AI Insights",  # Replace with the target room ID
        "text": f"The following alarm was received in Solarwinds from Node: \n {node} with IP Address: {ip_address}:\n\n {question}\n\n****The following is the recommended steps from Chatgpt:****\n\n {response} \n\n -------------------------END OF RECOMMENDATION--------------------------\n\n" # The message from ChatGPT
    }

    response = requests.post(webex_api_url, headers=headers, json=data)
    print(json.dumps(data, indent=4))

    if response.status_code == 200:
        return "Message sent to Webex successfully"
    else:
        return "Error: Failed to send message to Webex"


def get_token():

    token_url = "https://apigw.truvista.biz/connect/token"
    payload = {
        'client_id': 'Cisco-Alarms_8bc774efb54441cf8b201ff06feb591c',
        'client_secret': '7C',
        'grant_type': 'client_credentials',
        'scopes': 'eapi'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.request("POST", token_url, headers=headers, data=payload)
    response_data = response.json()
    access_token = response_data['access_token']
    print(access_token)
    return access_token

def create_incident(message,node,ip_address,chatgpt_response,access_token):


    incident_url = "https://apigw.truvista.biz/CHR.API/Customers.DynamicsCRM.Incidents/incidents"

    payload = json.dumps({
        "title": message,
        "customerid_account@odata.bind": "/accounts(F6F6C1)",
        "chr_TroubleTypeId@odata.bind": "/chr_troubletypes(800AD999)",
        "chr_ReportedTroubleId@odata.bind": "/chr_reportedtroubles(45C7F8BB-)",
        "prioritycode": "2",
        "chr_troubletickettypes": "126770003",
        "chr_preferredcommunicationmethod": "126770000",
        "chr_contactemail": "ctcco@truvista.biz",
        "description": f" \n Node = {node}, \n IP={ip_address}, \n Syslog: {message}, \n\n ChatGTP recommendations: {chatgpt_response}",
        "chr_ServiceLocation@odata.bind": "/chr_servicelocations(CC850FF9-)"
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    response_data = requests.request("POST", incident_url, headers=headers, data=payload)
    print(response_data)
    print("Incident Opened")


#----------webhook to receive alarm from Solwarinds, ask chatgpt, send to Webex-----------------

@app.route('/webhook', methods=['POST'])
def solarwinds_alert():
    # Parse the incoming JSON alert from SolarWinds
    alert_data = request.json


    message = alert_data['syslog']
    #print(alert_data['IP'])
    #print(alert_data['NodeName'])
    #print(alert_data['syslog'])
    ip_address = (alert_data['IP'])
    node = (alert_data['NodeName'])

#------------send alarm to chatGPT-------------------------------------------

    # Send the user's message to ChatGPT
    chatgpt_response = send_to_chatgpt(message)

    # Store the ChatGPT question and response
    chatgpt_responses.append({
        "question": message,
        "response": chatgpt_response
    })

#------------send chatGPT insights to webex-----------------------------------

    # Send the ChatGPT response to Webex
    send_to_webex(message, ip_address, node, chatgpt_response)


#------------GET CHR Bearer Token---------------------------------------------

    response_data = get_token()
    access_token = response_data

#------------Create Incident in CHR---------------------------------------------

    response_data = create_incident(message,node,ip_address,chatgpt_response,access_token)

    #Send a response to SolarWinds
    response_data = {"status": "Alert received and processed"}
    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)(debug=True)
