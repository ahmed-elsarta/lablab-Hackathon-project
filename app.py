# Interaction with Humans
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Kor
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
import json


openai_api_key = 'your_api_key'


llm = ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key)
llm2 = OpenAI(model_name="text-davinci-003", openai_api_key= openai_api_key)



def printOutput(output):
    print(json.dumps(output,sort_keys=True, indent=3))




# Extract Medical Information From the User's Message
def data_extraction(input_text):

    Medical_Info_Schema = Object(
        
        # This what will appear in your output. It's what the fields below will be nested under.
        # It should be the parent of the fields below. Usually it's singular (not plural)
        id="medical_symptoms",
        
        # Natural language description about your object
        description="Medical information about a person",
        
        # Fields you'd like to capture from a piece of text about your object.
        attributes=[
            
            Number(
                
                id="systolic_blood_pressure",
                description="The systolic blood pressure of a person.",
            ),
            

            Number(
                
                id="diastolic_blood_pressure",
                description="The diastolic blood pressure of a person.",
            ),
            

            Text(
                
                id="sex",
                description="The sex of a person.",
            ),
            
            Number(
                id="age",
                description="The age of a person."
            ),

            Number(
                id="height",
                description="The height of a person."
            ),

            Number(
                id="weight",
                description="The weight of a person."
            )

            ,

            Number(
                id="temperature",
                description="The temperature of a person."
            )
            ,

            Number(
                id="heart_rate",
                description="The heart rate of a person."
            ),

            Number(
                id="cholesterol",
                description="The cholesterol level of a person."
            ),

            Number(
                id="glucose",
                description="The glucose level of a person."
            ),
            

            Text(
                
                id="activity",
                description="Is the person active or not.",
            ),
            

            Text(
                
                id="smoker",
                description="Is the person a smoker or not.",
            ),
            

            Text(
                
                id="alcohol",
                description="Does the person drink alcohol or not.",
            ),

        ],
        

        # Examples help go a long way with telling the LLM what you need
        examples = [(
            
            '''I am a 22 year old girl, I am 166 cm tall and weigh 60 kg my blood pressure is 119/80, body temperature is 37 and heart rate is 89 bpm, I walk 30 minutes every morning
            and don't smoke nor drink alcohol, my glucose and cholesterol levels were 0.2 and 0.5 respectively
            ''', 
            
            [{ "systolic_blood_pressure": 119, "diastolic_blood_pressure": 80, "sex": "Female", "age": 22 , "height": 166, "weight": 60, "temperature": 37, "heart_rate": 89, "cholesterol": 0.5, "glucose": 0.2, 
                "activity": "yes", "smoker": "no", "alcohol": "no"}], 

            )]
    )


    print(input_text)
    chain = create_extraction_chain(llm, Medical_Info_Schema)
    output = chain.predict_and_parse(text= input_text)['data']

    printOutput(output)
    print(type(output))

    return output







# Check that all the required information was given
def check_information(input_text, gender):

    template = """You are an AI bot that would keep asking the person till he or she gives you all the required medical information, you 
            can't help in any other field, the amount of information would differ whether it is a female or male gender.

            You should respond to the user in the same language, for example if the user messages you in arabic, you should respond
            in arabic not in another language.
    
            If the gender is male only and not female the following attributes shall be given in the input text and if one of them is not given for a male keep asking  for the missing
            attributes till they are given, 
            the attributes for a male gender are, blood_pressure, sex, age, height, weight, cholesterole and glucose levels,
            rate of activity, whether a smoker or alcohol drinker or not. 

            If the gender is female only and not male the following 5 attributes only shall be given in the input text, and if one of them is not given for a female keep asking for the
            missing attributes till they are given        
            the attributes for a female gender are, age, blood pressure, body temperature, heart rate and glucose level only, 
            no cholesterol level or weight needed.
            
            User Input:
            {input_text}

            Gender:
            {gender}

            RESPONSE:  
            """


    prompt = PromptTemplate(input_variables=["input_text", "gender"], template= template)
        
    final_prompt = prompt.format(input_text= f'{input_text}', gender= f'{gender}')

    print (f"Final Prompt: {final_prompt}")
    print (f"{llm2(final_prompt)}")

    return llm2(final_prompt)







# Compute the risk to a certain disease
def risk_assessment(risk_number, gender):

    template = """ You will be given a message from the user in number format, where number 1 represents high risk, number 0 represents low risk and number 0.5 represents medium risk.
    Reply to the user with low or high or medium risk of carrdiovascular disease if it is a male
    or low or high risk of maternal disease if it is a female according to the given number and gender
    and make sure to leave some suggestions in friendly tone


    Risk Value:
    {risk_number}

    Gender:
    {gender}

    YOUR RESPONSE:
    """

    prompt = PromptTemplate(
    input_variables=["risk_number", "gender"],
    template= template,
    )

    final_prompt = prompt.format(risk_number= f'{risk_number}', gender=f'{gender}')

    print (f"Final Prompt: {final_prompt}")
    print (f"{llm2(final_prompt)}")

    return llm2(final_prompt)
