# Interaction with Humans
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Kor
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
import json


openai_api_key='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'


llm = ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key)
llm2 = OpenAI(model_name="text-davinci-003", openai_api_key= openai_api_key)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm = llm2, memory = memory)



def printOutput(output):
    print(json.dumps(output,sort_keys=True, indent=3))




# Extract Medical Information From the User's Message (prompt)
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

            ), (
            
            '''I am a 50 year old female, I am 166 cm tall and weigh 60 kg my blood pressure is 119/80, body temperature is 37 and heart rate is 89 bpm, I walk 30 minutes every morning
            and don't smoke nor drink alcohol, my glucose and cholesterol levels were 0.2 and 0.5 respectively
            ''', 
            
            [{ "systolic_blood_pressure": 119, "diastolic_blood_pressure": 80, "sex": "Female", "age": 50, "height": 166, "weight": 60, "temperature": 37, "heart_rate": 89, "cholesterol": 0.5, "glucose": 0.2, 
                "activity": "yes", "smoker": "no", "alcohol": "no"}], 

            ), ( 
            
            '''I am a 30 year old male, I am 166 cm tall and weigh 60 kg my blood pressure is 119/80, body temperature is 37 and heart rate is 89 bpm, I like staying at home
            and don't smoke nor drink alcohol, my glucose and cholesterol levels were 0.2 and 0.5 respectively
            ''', 
            
            [{ "systolic_blood_pressure": 119, "diastolic_blood_pressure": 80, "sex": "Male", "age": 30 , "height": 166, "weight": 60, "temperature": 37, "heart_rate": 89, "cholesterol": 0.5, "glucose": 0.2, 
                "activity": "no", "smoker": "no", "alcohol": "no"}],)

        ]

    )

    print(input_text)
    chain = create_extraction_chain(llm2, Medical_Info_Schema)
    output = chain.predict_and_parse(text= input_text)['data']

    printOutput(output)
    print(type(output))

    return output






'''def check():
    answer = conversation.predict(input = f'''#Is all the medical information available for me from our conversation, 
                           # reply with Yes or No only, keep them in the same format''') 
    #memory.save_context({"input": f"{user_input}"}, {"output": f"{response}"})
    #return answer'''

'''that would keep asking the person till he or she gives you all the required medical 
            information only, and you can't help in any other field, 
            the amount of information given to you would differ whether it is a female or male gender.'''



# Check that all the required information was given
def check_information(input_text, gender, AI_response):

    template = """You are an AI bot that would keep asking the person till he or she gives you all the required medical information only,
               you can't help in any other field except the medical field to provide risk assessment for the cardiovascular diseases or the maternal diseases,
               the amount of information would differ whether it is a female or male gender.

            You have a minimum set of informations that need to be given to you from the user for the risk assessment based on the 
            gender of the user, whether male or female.

            You should respond to the user in the same language, for example if the user messages you in arabic, you should respond
            in arabic not in another language.

            If the given gender was Male, then the user's gender is male , if the given gender was Female, then the user's gender is female.

            If the user's gender is male and not female the following attributes only have to be given in the input text
            and if one of them is not given for a male keep asking  for the missing attributes till they are given, 
            the attributes for a male gender are, blood_pressure, age, height, weight, cholesterole and glucose levels,
            rate of activity, whether a smoker or alcohol drinker or not. 

            If the user's gender is female and not male the following 5 attributes only have to be given in the input text
            , and if one of them is not given for a female keep asking for the missing attributes till they are given    
            the attributes for a female gender are, age, blood pressure, body temperature, heart rate and glucose level only, 
            no cholesterol level or weight needed.

            Please make sure that all information is present for the relevant gender this is very important.

            If you know the gender of the user ask directly for the medical information.

            You are allowed to chat with the user but in the cardiovascular or maternal medical fields only.

            
            USER_INPUT:
            {input_text}


            GENDER:
            {gender}

            YOUR RESPONSE:
            """


    prompt = PromptTemplate(input_variables=["input_text", "gender"], template= template)      
    final_prompt = prompt.format(input_text= f'{input_text}', gender= f'{gender}')

    print (f"Final Prompt: {final_prompt}")
    print (f"{llm2(final_prompt)}")

    # Save the context to the memory
    memory.save_context({"input": f"{input_text}"}, {"output": f"{llm2(final_prompt)}"})
    response = llm2(final_prompt)
    #response = conversation.predict(input = template)

    '''prompt = PromptTemplate(input_variables=["input_text", "gender"], template= template)      
    final_prompt = prompt.format(input_text= f'{input_text}', gender= f'{gender}') '''
    return response
    







# Compute the risk to a certain disease from the pre-made model
def risk_assessment(input_text, risk_number, gender):

    template = """ You will be given a message from the user in number format, where number 1 represents high risk, number 0 represents 
    low risk and number 0.5 represents medium risk.
    Reply to the user with low or high or medium risk of carrdiovascular disease if it is a male
    or low or high risk of maternal disease if it is a female according to the given number and gender
    and make sure to leave some suggestions in friendly tone and don't include numbers in your response


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
    # Save the context to the memory
    memory.save_context({"input": f"{input_text}"}, {"output": f"{llm2(final_prompt)}"})
    response = llm2(final_prompt)

    print (f"Final Prompt: {final_prompt}")
    print (f"{llm2(final_prompt)}")

    return response





# Get the gender from old chat
def get_gender(user_input, AI_response):
    memory.save_context({"input": f"{user_input}"}, {"output": f"{AI_response}"})
    gender = conversation.predict(input = '''Get my gender from my name or if I gave it to you directly or from our past conversation,
    reply with one word only as Female or Male only and keep the same format for them''')
    
    return gender


