from flask import Flask, request, jsonify

from my_generate import UserResponseGenerator


app = Flask(__name__)

generate_model = UserResponseGenerator("karakuri-ai/karakuri-lm-8x7b-chat-v0.1")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()

    model_name = data.get('model_name')
    conversations = data.get('conversations', [])

    response_data = generate_model.generate_user_response(model_name, conversations)

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=13513)
