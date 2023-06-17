import streamlit as st
import joblib
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat
from hugchat.login import Login
import app



# Empty dictionary for extracted data
keys1 = ["age", "sex", "weight", "systolic_blood_pressure", "diastolic_blood_pressure", "cholesterol","glucose", 
         "smoker", "alcohol", "activity"]
         
keys2 = ["age", "systolic_blood_pressure", "diastolic_blood_pressure", "glucose", "temperature", "heart_rate"]
       

risk_dict = {"low risk": 0, "high risk": 1, "mid risk": 0.5}
string_to_num_dict= {"yes": 1, "no": 0}

male_data = {key: None for key in keys1}
female_data = {key: None for key in keys2}
output_dict = male_data
state = 0
gender="null"
ai_response ="Hi"
risk_val = 0


# Load the saved model for cardiovascular disease from disk
model_male = joblib.load("D:/LangChain - Project/notebook/model.sav")

# Load the saved model for maternal disease from disk
model_female = joblib.load("D:/LangChain - Project/notebook/maternal_model.sav")



# Log in to huggingface and grant authorization to huggingchat
sign = Login("saraayman10000@gmail.com", "Hugging_sara123")
cookies = sign.login()
# Save cookies to usercookies/<email>.json
sign.saveCookies()



st.set_page_config(page_title="Nurse GPT - An LLM-powered Streamlit app")


# Sidebar contents
with st.sidebar:
    st.title(':white_check_mark: Nurse GPT')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [HugChat](https://github.com/Soulter/hugging-chat-api)
    ''')
    add_vertical_space(5)



# Generate empty lists for generated and past.
## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']


# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()


# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()



# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    
    global state, gender, ai_response, risk_val, female_data, male_data
    # Create a ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  #response = chatbot.chat(prompt)
    
    output_dict = app.data_extraction(prompt)   # Get the data from user's text
    print(output_dict["medical_symptoms"])
    # If the user didn't give any useful information ask the user again for info
    if not bool(output_dict["medical_symptoms"]):
        ai_response = app.check_information(prompt, "unknown", ai_response)
        gender = app.get_gender(prompt, ai_response)
        #print("The dictionary is empty")
        return ai_response

    else:
        # The first trial for the user
        if state==0:

            state = 1
            #female_data = output_dict["medical_symptoms"][0]
            for key, value in female_data.items():  

                value = output_dict["medical_symptoms"][0][key]
                if (value) and (value!= "unknown") and (value!= "null") and (value!= "N/A"):
                    female_data[key] = value

            # if the value was null, ask the user for more info.
            for key, value in female_data.items():   
                
                if (not bool(value)) or (value == "null") or (value == "unknown") or (value == "N/A"): 
                    ai_response = app.check_information(prompt, "unknown", ai_response)
                    gender = app.get_gender(prompt, ai_response)
                    return ai_response   

            state = 0
            # Get the data vector to make a prediction on
            data = [[ female_data["age"] , female_data["systolic_blood_pressure"], female_data["diastolic_blood_pressure"]
                                , female_data["glucose"], female_data["temperature"], female_data["heart_rate"] ]]
            # Make a prediction using the loaded model
            risk_val = model_female.predict(data)
            ai_response = app.risk_assessment(prompt, risk_dict[risk_val[0]], gender)
            gender = app.get_gender(prompt, ai_response)
            print(state)
            female_data = {key: None for key in keys2}  #print(female_data)   #print(risk_val)
            return ai_response
            

        # Get Missing Data from the user
        elif state == 1:

            for key, value in female_data.items():                        
                value = output_dict["medical_symptoms"][0][key]
                # If the value wasn't null copy to female data array
                if (value) and (value!="unknown") and (value!="null") and (value!= "N/A"):
                    female_data[key] = output_dict["medical_symptoms"][0][key]

            state = 0
            # Get the data vector to make a prediction on
            data = [[ female_data["age"] , female_data["systolic_blood_pressure"], female_data["diastolic_blood_pressure"]
                                , female_data["glucose"], female_data["temperature"], female_data["heart_rate"] ]]
            risk_val = model_female.predict(data)
            ai_response = app.risk_assessment(prompt, risk_dict[risk_val[0]], gender)
            gender = app.get_gender(prompt, ai_response)
            print(state)
            print(female_data)
            female_data = {key: None for key in keys2}
            return ai_response   
        

                
        # If the gender is male perform the following
        if gender==" Male":

            # The first trial for the user
            if state==0:

                state = 1
                for key, value in male_data.items():  
                    value = output_dict["medical_symptoms"][0][key]
                    if (value) and (value!="unknown") and (value!= "null") and (value!= "N/A"):
                        male_data[key] = value

                # if the value was null, ask the user for more info.
                for key, value in male_data.items():   
                    if (not bool(value)) or (value =="null") or (value=="unknown") or (value== "N/A"): 
                        ai_response = app.check_information(prompt, "unknown", ai_response)
                        gender = app.get_gender(prompt, ai_response)
                        return ai_response   

                state = 0
                # Get the data vector to make a prediction on
                data = [[ male_data["age"], 0, male_data["weight"],
                        male_data["systolic_blood_pressure"], male_data["diastolic_blood_pressure"],male_data["cholesterol"], 
                        male_data["glucose"], string_to_num_dict[male_data["smoker"]], string_to_num_dict[male_data["alcohol"]]
                        , string_to_num_dict[male_data["activity"]] ]]
                # Make a prediction using the loaded model
                risk_val = model_male.predict(data)
                ai_response = app.risk_assessment(prompt, risk_val, gender)
                gender = app.get_gender(prompt, ai_response)
                male_data = {key: None for key in keys1}
                print(state)
                return ai_response
            
            # Get Missing Data from the user
            elif state == 1:

                for key, value in male_data.items():                        
                    value = output_dict["medical_symptoms"][0][key]
                    # If the value wasn't null copy to male data array
                    if (value) and (value!="unknown") and (value!="null") and (value!= "N/A"):
                        male_data[key] = output_dict["medical_symptoms"][0][key]

                state = 0
                # Get the data vector to make a prediction on
                data = [[ male_data["age"], 0, male_data["weight"],
                        male_data["systolic_blood_pressure"], male_data["diastolic_blood_pressure"],male_data["cholesterol"], 
                        male_data["glucose"], string_to_num_dict[male_data["smoker"]], string_to_num_dict[male_data["alcohol"]]
                        , string_to_num_dict[male_data["activity"]] ]]
                risk_val = model_female.predict(data)
                ai_response = app.risk_assessment(prompt, risk_val, gender)
                gender = app.get_gender(prompt, ai_response)
                male_data = {key: None for key in keys1}
                return ai_response   


    #print("Sheeeeeeeeesh")
    #print(state)
    ai_response = app.check_information(prompt, "unknown", ai_response)
    gender = app.get_gender(prompt, ai_response)
    return ai_response





## Conditional display of AI generated responses as a function of user provided prompts
with response_container:

    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
