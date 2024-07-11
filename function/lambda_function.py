#!/bin/python3
"""
This is a Slack app that uses the Slack Events API and
AWS Lambda to provide a Slack bot that can be used to
answer questions based on articles in a knowledge base.
"""
import json
import re
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from slack_sdk.errors import SlackApiError

app = App(process_before_response=True)


def send_message_to_user(user_id, message):
    try:
        # Open a direct message channel with the user
        response = app.client.conversations_open(users=user_id)
        channel_id = response['channel']['id']  # type: ignore

        # Send the message to the user
        app.client.chat_postMessage(channel=channel_id, text=message)

    except SlackApiError as e:
        print(f"Error sending message: {e}")


def extract_valid_json(string:str) -> dict:
    string_clean = string.replace('\n', '')
    # Regex pattern for basic JSON structure: objects {} and arrays []
    json_pattern = re.compile(r'\{.*?\}|\[.*?\]', re.DOTALL)
    # Finding all matches that look like JSON
    potential_jsons = json_pattern.findall(string_clean)
    if not potential_jsons:
        return None
    for pj in potential_jsons:
        try:
            # Attempt to parse the JSON
            valid_json = json.loads(pj)
            # Returning the first valid JSON found
            return valid_json
        except json.JSONDecodeError:
            continue
    return None


# @app.event('message')
# def handle_message_events(body, say):
#     print("Received message event")
        
#     event = body['event']
#     message = event['text']
#     say(f'Question: {message}')
    
#     # Only importing here to optimize Lambda start up time
#     from completion import get_completion
#     from embedding import get_data, get_document_embeddings, construct_prompt

#     # It's possible that the message is from a bot user.
#     # If so, we don't want to respond to it or it might cause
#     # a infinite loop - and putting your AWS bill on fire!
#     is_user_message = (
#         event['type'] == 'message' and
#         (not event.get('subtype')) and
#         (not event.get('bot_id'))
#     )
#     if not is_user_message:
#         return

#     print(f'User said `{message}`.')

#     df = get_data()
#     embeddings = get_document_embeddings()
#     prompt_with_articles_as_context = construct_prompt(message, embeddings, df)
#     reply_text = get_completion(prompt_with_articles_as_context)
#     say(reply_text)


# @app.event('message')
# def handle_message_events(body, say):
#     print("Received message event")
        
#     event = body['event']
#     message = event['text']
    
#     # Only importing here to optimize Lambda start up time
#     from completion import get_completion

#     # It's possible that the message is from a bot user.
#     # If so, we don't want to respond to it or it might cause
#     # a infinite loop - and putting your AWS bill on fire!
#     is_user_message = (
#         event['type'] == 'message' and
#         (not event.get('subtype')) and
#         (not event.get('bot_id'))
#     )
#     if not is_user_message:
#         return

#     glossary = {
#         'RAG': 'Retrival Augmented Generation',
#         'GPT': 'Generative Pre-trained Transformer',
#         'API': 'Application Programming Interface',
#         'ML': 'Machine Learning',
#         'NLP': 'Natural Language Processing',
#         'AI': 'Artificial Intelligence',
#         'DL': 'Deep Learning',
#         'lmao': 'laughing my ass off',
#         'AWS': 'Amazon Web Servies',
#         'GCP': 'Google Cloud Platform',
#         'GTM': 'Go To Market',
#         'TAM': 'Total Addressable Market',
#     }
#     prompt = f"""
# #GLOSSARY
# {glossary}

# #USER MESSAGES
# {message}

# #TASK
# If the USER MESSAGE contains any anroynyms of buzz-words in the GLOSSARY or that might be confusing for someone else, then populate the JSON with their meanings, otherwise return an empty JSON. Example:
# {{"RAG": "Retrival Augmented Generation", "DWP": "Department for Work and Pensions"}}
# """
#     response = get_completion(prompt)
#     clean_json = extract_valid_json(response)
#     if clean_json is None or len(clean_json) == 0:
#         say()
#     else:
#         str_output = '\n'.join([f"• *{term}* => _{meaning}_" for term, meaning in clean_json.items()])
#         say(f"Please mind the following terms stand for:\n{str_output}")


@app.event('message')
def handle_message_events(body, say, ack):
    print("Received message event")
        
    event = body['event']
    message = event['text']
    thread_ts = event.get('thread_ts') or event['ts']
    
    # Only importing here to optimize Lambda start up time
    from embedding import get_glossary

    glossary_df = get_glossary().reset_index()
    print('Glossary df (reset index):', glossary_df)
    # convert the glossary dataframe to a dictionary
    glossary_dict = glossary_df.set_index('term')['meaning'].to_dict()
    glossary_terms = glossary_dict.keys()

    glossary_terms_in_message = {}
    for term in glossary_terms:
        pattern = r'(?<![\w-])' + re.escape(term) + r'(?![\w-])'
        match = re.search(pattern, message)
        if match:
            glossary_terms_in_message[term] = glossary_dict[term]

    print('Glossary terms in message:', glossary_terms_in_message)
    if len(glossary_terms_in_message) == 0:
        say()
    else:
        str_output = '\n'.join([f"• *{term}* => _{meaning}_" 
                                for term, meaning in glossary_terms_in_message.items()])
        say(f"Please mind the following terms stand for:\n{str_output}", thread_ts=thread_ts)
        ack()
    
    
@app.command("/genny")
def handle_gennyai_command(ack, body, say):
    ack()
    print('Recieved /genny command')

    message = body['text']
    print(f'User said `{message}`.')
    say(f'Question: {message}')
    
    try:
        # Only importing here to optimize Lambda start up time
        from completion import get_completion
        from embedding import get_data, get_document_embeddings, construct_prompt
    
        print('body:', body)
        df = get_data()
        embeddings = get_document_embeddings()
        prompt_with_articles_as_context = construct_prompt(message, embeddings, df)
        reply_text = get_completion(prompt_with_articles_as_context + '. Respond in 40 words or less')
        say(reply_text)
    except Exception as e:
        print(f'Error getting result {e}')


@app.command('/genny-faq')
def handle_submit_train_article_command(ack, respond, command):
    ack()
    print('Received /genny-faq command')
    try:
        _response = app.client.views_open(
            trigger_id=command['trigger_id'],
            view={
                'type': 'modal',
                'callback_id': 'gennyai-faq_view',
                'title': {'type': 'plain_text', 'text': "Add new FAQ"},
                'blocks': [
                    {
                        'type': 'input',
                        'block_id': 'title_block',
                        'label': {'type': 'plain_text', 'text': 'Title'},
                        'element': {'type': 'plain_text_input', 'action_id': 'title_input'},
                    },
                    {
                        'type': 'input',
                        'block_id': 'heading_block',
                        'label': {'type': 'plain_text', 'text': 'Heading'},
                        'element': {'type': 'plain_text_input', 'action_id': 'heading_input'},
                    },
                    {
                        'type': 'input',
                        'block_id': 'content_block',
                        'label': {'type': 'plain_text', 'text': 'Content'},
                        'element': {'type': 'plain_text_input', 'multiline': True, 'action_id': 'content_input'},
                    },
                    {
                        'type': 'actions',
                        'block_id': 'delete_action_block',
                        'elements': [
                            {
                                'type': 'button',
                                'text': {'type': 'plain_text', 'text': 'Advanced: View/Edit'},
                                'action_id': 'delete_article_btn',
                            }
                        ]
                    },
                ],
                'submit': {'type': 'plain_text', 'text': 'Submit'},
            },
        )
    except Exception as e:
        print('Error opening modal: {}'.format(e))


@app.view('gennyai-faq_view')
def handle_new_train_article_submission(ack, body):
    ack()
    values = body['view']['state']['values']

    title = values['title_block']['title_input']['value']
    heading = values['heading_block']['heading_input']['value']
    content = values['content_block']['content_input']['value']

    print(f'Adding training data: {title}, {heading}, {content}')

    # "Lazy loading" to avoid long Lambda start up times
    from embedding import process_new_article
    result = process_new_article(title, heading, content)
    ack(response_action='clear')

    if result:
        # send message with Slack client `client`
        send_message_to_user(
            body['user']['id'],
            f'New training data added: with index ({title}, {heading})',
        )
        print(f'New training data added! {title}, {heading}')
    else:
        send_message_to_user(
            body['user']['id'],
            'Something went wrong when adding new training data!',
        )
        print('Something went wrong when adding new training data!')
    

@app.action('delete_article_btn')
def handle_delete_article_button_click(ack, body, client):
    ack()
    print('Received delete article button click.')

    from embedding import get_data, get_datafile_presigned_url
    data = get_data()
    print('data', data)
    s3_path, datafile_url = get_datafile_presigned_url()
    articles = data.reset_index().to_dict('records')
    print(f'articles: {articles}')
    article_options = [{
        'text': {
            'type': 'plain_text', 
            'text': f"{article['title']} | {article['heading']}"
        },
        'value': str(idx)
        }
        for idx, article in enumerate(articles)
    ]
    print(f'article_options: {article_options}')

    try:
        print('Updating modal to show delete options...')
        client.views_update(
            view_id=body["view"]["id"],
            view={
                "type": "modal",
                "callback_id": "delete_article_view",
                "title": {"type": "plain_text", "text": "Delete Article"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "Select an article to delete:"},
                        "accessory": {
                            "type": "multi_static_select",
                            "placeholder": {"type": "plain_text", "text": "Select an article"},
                            "action_id": "article_select",
                            "options": article_options,
                            "max_selected_items": 10
                        }
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"<{datafile_url}|Download Data>"},
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "plain_text",
                                "text": f"R&D Sandbox ({s3_path})",
                                "emoji": True
                            }
                        ]
                    }
                ],
                "submit": {"type": "plain_text", "text": "Delete"},
            }
        )
        print('Delete article modal updated.')
    except Exception as e:
        print(f"Error updating delete article modal: {e}")


@app.view('delete_article_view')
def handle_article_deletion(ack, body, client):
    ack()
    print('Received delete article submission.')
    print('body:', body)
    block_id = body['view']['blocks'][0]['block_id']
    selected_articles = body['view']['state']['values'][block_id]['article_select']['selected_options']
    selected_articles_title_heading = [tuple(article['text']['text'].split(' | '))
                                       for article in selected_articles]
    print(f'Selected articles: {selected_articles_title_heading}')

    # "Lazy loading" to avoid long Lambda start up times
    from embedding import delete_article
    result = delete_article(selected_articles_title_heading)  # Implement this function to delete the article
    ack(response_action='clear')

    if result:
        # send message with Slack client `client`
        send_message_to_user(
            body['user']['id'],
            f'Articles sucessfully deleted: {selected_articles_title_heading}',
        )
        print(f'Articles deleted! {selected_articles_title_heading}')
    else:
        send_message_to_user(
            body['user']['id'],
            'Something went wrong deleting articles!',
        )
        print('Something went wrong deleting articles!')


@app.command('/genny-glossary')
def handle_submit_glossary_command(ack, respond, command):
    ack()
    print('Received /genny-glossary command')
    try:
        _response = app.client.views_open(
            trigger_id=command['trigger_id'],
            view={
                'type': 'modal',
                'callback_id': 'genny-glossary_view',
                'title': {'type': 'plain_text', 'text': 'Add new Glossary terms'},
                'blocks': [
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "plain_text",
                                "text": "Example Content:\n\nRAG=Retrieval Augmented Generation\nAI=Artificial Intelligence",
                                "emoji": True
                            }
                        ]
                    },
                    {
                        'type': 'input',
                        'block_id': 'content_block',
                        'label': {'type': 'plain_text', 'text': 'Content'},
                        'element': {'type': 'plain_text_input', 'multiline': True, 'action_id': 'content_input'},
                    },
                    {
                        'type': 'actions',
                        'block_id': 'delete_glossary_block',
                        'elements': [
                            {
                                'type': 'button',
                                'text': {'type': 'plain_text', 'text': 'Advanced: View/Edit'},
                                'action_id': 'delete_glossary_btn',
                            }
                        ]
                    },
                ],
                'submit': {'type': 'plain_text', 'text': 'Submit'},
            },
        )
    except Exception as e:
        print('Error opening modal: {}'.format(e))


@app.view('genny-glossary_view')
def handle_new_glossary_submission(ack, body):
    ack()
    values = body['view']['state']['values']
    terms_meanings = []
    for values in values['content_block']['content_input']['value'].split('\n'):
        term, meaning = values.split('=')
        if len(term.split(' ')) > 3:
            continue
        terms_meanings.append((term.strip(), meaning.strip()))

    print(f'Adding glossary data: {terms_meanings}')

    # "Lazy loading" to avoid long Lambda start up times
    from embedding import process_new_glossary
    result = process_new_glossary(terms_meanings)
    ack(response_action='clear')

    if result:
        # send message with Slack client `client`
        send_message_to_user(
            body['user']['id'],
            f'New glossary data added: with index ({terms_meanings})',
        )
        print(f'New glossary data added! {terms_meanings}')
    else:
        send_message_to_user(
            body['user']['id'],
            'Something went wrong when adding new glossary data!',
        )
        print('Something went wrong when adding new glossary data!')


@app.action('delete_glossary_btn')
def handle_delete_glossary_button_click(ack, body, client):
    ack()
    print('Received delete glossary button click.')

    from embedding import get_glossary, get_glossary_presigned_url
    glossary = get_glossary()
    print('glossary', glossary)
    s3_path, glossaryfile_url = get_glossary_presigned_url()
    glossary_terms = glossary.reset_index().to_dict('records')
    print(f'glossary_terms: {glossary_terms}')
    glossary_options = [{
        'text': {
            'type': 'plain_text', 
            'text': f"{glossary_term['term']} | {glossary_term['meaning']}"
        },
        'value': str(idx)
        }
        for idx, glossary_term in enumerate(glossary_terms)
    ]
    print(f'glossary_terms: {glossary_terms}')

    try:
        print('Updating modal to show delete glossary options...')
        client.views_update(
            view_id=body["view"]["id"],
            view={
                "type": "modal",
                "callback_id": "delete_glossary_view",
                "title": {"type": "plain_text", "text": "Delete Glossary Term"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "Select an Glossary term to delete:"},
                        "accessory": {
                            "type": "multi_static_select",
                            "placeholder": {"type": "plain_text", "text": "Select an glossary entry"},
                            "action_id": "glossary_select",
                            "options": glossary_options,
                            "max_selected_items": 10
                        }
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"<{glossaryfile_url}|Download Data>"},
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "plain_text",
                                "text": f"R&D Sandbox ({s3_path})",
                                "emoji": True
                            }
                        ]
                    }
                ],
                "submit": {"type": "plain_text", "text": "Delete"},
            }
        )
        print('Delete glossary term modal updated.')
    except Exception as e:
        print(f"Error updating delete glossary term modal: {e}")


@app.view('delete_glossary_view')
def handle_article_deletion(ack, body, client):
    ack()
    print('Received delete glossary term submission.')
    print('body:', body)
    block_id = body['view']['blocks'][0]['block_id']
    selected_glossary = body['view']['state']['values'][block_id]['glossary_select']['selected_options']
    selected_glossary_term_meanings = [tuple(value['text']['text'].split(' | '))
                                      for value in selected_glossary]
    print(f'Selected glossary terms: {selected_glossary_term_meanings}')

    # "Lazy loading" to avoid long Lambda start up times
    from embedding import delete_glossary
    result = delete_glossary(selected_glossary_term_meanings)  # Implement this function to delete glossary term
    ack(response_action='clear')

    if result:
        # send message with Slack client `client`
        send_message_to_user(
            body['user']['id'],
            f'Glossary terms sucessfully deleted: {selected_glossary_term_meanings}',
        )
        print(f'Glossary terms deleted! {selected_glossary_term_meanings}')
    else:
        send_message_to_user(
            body['user']['id'],
            'Something went wrong deleting glossary terms!',
        )
        print('Something went wrong deleting glossary terms!')



def lambda_handler(event, context):
    # Sometimes Slack might fire a 2nd attempt, etc
    # if the first attempt fails or expires after some time.
    # In that case we can try to achieve idempotency by checking
    # the X-Slack-Retry-Num and skip processing the event, so that
    # only the first attempt is processed.
    # Ref: https://api.slack.com/apis/connections/events-api#retries

    print('made it to lambda_handler')
    if event['headers'].get('x-slack-retry-num'):
        print('Retry attempt detected. Skipping processing.')
        return {
            'statusCode': 200,
            'body': 'ok',
        }

    # For debugging
    print(json.dumps(event))

    try:
        # Extract the body from the event
        body = json.loads(event['body'])
        if body['type'] == 'url_verification':
            challenge = body['challenge']
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'text/plain'},
                'body': challenge
            }
    except Exception as e:
        print(f'No challenge token found. {e}')
    
    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)
