from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

WEBEX_BOT_TOKEN = ''
CHATGPT_API_KEY = 'sk-'
ALLOWED_EMAIL_DOMAINS = ["truvista.biz"]


###################################################################

def get_message_details(message_id):
    url = f"https://webexapis.com/v1/messages/{message_id}"
    headers = {
        "Authorization": f"Bearer {WEBEX_BOT_TOKEN}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['text']
    else:
        return None


###################################################################

def clean_message(message, bot_trigger):
    """
    Remove the bot trigger from the message.

    :param message: The raw message text received.
    :param bot_trigger: The trigger or mention used for the bot.
    :return: Cleaned message with the bot trigger removed.
    """
    if message.startswith(bot_trigger):
        return message[len(bot_trigger):].strip()
    return message


###################################################################

def send_interim_message_to_webex(room_id):
    interim_message = "Hang tight, while I check into that for you."
    send_message_to_webex(room_id, interim_message)

###################################################################

def send_message_to_webex(room_id, message):
    url = "https://webexapis.com/v1/messages"
    headers = {
        "Authorization": f"Bearer {WEBEX_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "roomId": room_id,
        "text": message
    }
    response = requests.post(url, headers=headers, json=payload)
    return response

###################################################################

def ask_chatgpt(question):
    chatgpt_api_url = "https://api.openai.com/v1/chat/completions"
    chatgpt_api_key = "sk-"

    headers = {
        "Authorization": f"Bearer {chatgpt_api_key}",
        "Content-Type": "application/json"
    }


    data = {
        "model": "gpt-4-0125-preview",
        "messages": [{"role": "system", "content": "You are ISP Network Expert, specializing in Cisco, Adtran, Telco Systems,  and Arris systems. "},
    {"role": "user", "content": question}]
    }

    response = requests.post(chatgpt_api_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return "Sorry, I couldn't process that question."

##################################################################

@app.route('/webhook', methods=['POST'])
def webhook():
    json_data = request.json

    # Extract messageId and roomId from the webhook data
    message_id = json_data['data']['id']
    room_id = json_data['data']['roomId']
    sender_email = json_data.get('data', {}).get('personEmail', '')
    sender_domain = sender_email.split('@')[-1] if '@' in sender_email else ''
    print("Received Webhook Data:", json_data)


    # Prevent the bot from responding to itself
    if json_data['data']['personEmail'] == 'askanetworkexpert@webex.bot':
        return jsonify({'message': 'Ignoring bot\'s own message'}), 200

    # Get the message details from Webex if email matches domain
    if sender_domain in ALLOWED_EMAIL_DOMAINS:
        message_text = get_message_details(message_id)
        print("Extracted Message:", message_text)
    else:
        return jsonify({'error': 'Unauthorized access'}), 403

    # Clean Message

    bot_trigger = "askanetworkexpert@webex.bot"
    message_text = clean_message(message_text, bot_trigger)

    # Ask chatGPT and Send message to webex

    if message_text:
        send_interim_message_to_webex(room_id)
        chatgpt_response = ask_chatgpt(message_text)
        send_message_to_webex(room_id, chatgpt_response)
        return jsonify({'message': 'Message sent to Webex'}), 200
    else:
        return jsonify({'message': 'Failed to retrieve message details'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)(debug=True)
