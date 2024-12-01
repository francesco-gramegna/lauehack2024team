from openai import OpenAI
import openai
import json

#Usage : event is the string passed by the chatgpt
# time is the time used to fill in the "time"
def getEventValuesJson(event, time)
    apikey = open("key","r").read()
    
    if apikey == "":
        print("no key?")
    apikey = apikey.strip()
    
    client = OpenAI(api_key=apikey)
    
    
    timeOfEvent = time
    actionType = "Set"
    
    
    vars = (open("vars/input.json").read()) #can be changed to be a function
    
    systemstring = """You need to respond in a json format. The prompt is an event, and you need to adapt these variables in order to match the event.
    These variables are medical information about a pharmaceutical company.
    Your response must contain for each field a "multiplicator" field (eg : 1.1 for +10%),
    a "duration" field that explains in how much time (in months) the change takes effect,
    and an explainaition of why.
    Variables : """ + vars
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": systemstring},
            {
    
            "role": "user",
            "content": event,
        }],
        model="gpt-4o",
    
        )
    
    
    jsondata = (response.choices[0].message.content).replace("```", "").replace("json", "").replace("multiplicator", "factor")
    
    print(jsondata)
    
    jsondata = json.loads(jsondata)
    for key in jsondata:
        jsondata[key]["time"] = timeOfEvent
    
    print(jsondata)
    
    jsondata

