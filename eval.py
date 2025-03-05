from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic
#from ragas.metrics import LLMContextRecall
#from ragas.metrics import LLMContextPrecisionWithoutReference
#from ragas.metrics import NoiseSensitivity
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from generation import generation

import asyncio


async def evaluate_test_data():
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))


    test_data = {
        "user_input": "What is warfighter function?",
        "response": "A warfighting function is a group of tasks and systems united by a common purpose that commanders use to accomplish missions and training objectives. Warfighting functions are the physical means that tactical commanders use to execute operations and accomplish missions assigned by superior tactical- and operational-level commanders. The purpose of warfighting functions is to provide an intellectual organization for common critical capabilities available to commanders and staffs at all echelons and levels of warfare."
    }

    test_data2 = [
        {
            "role": "system",
            "content": "You are a battalion executive and operations officer specialized in leading and conducting military decision-making process."
        },
        {
            "role": "user",
            "content": "I am an infantry battalion commander, and have been asked to secure a airstrip on Sicily Drop Zone, with a time on target of 0400 on 1 March, 2025. My battalion, 2-508th PIR, will be augmented with an SAPPER engineer platoon and M777 artillery battery. Draft a warning order, consistent with FM 5.0, in the 5 paragraph oporder format."
        }
    ]

    metric_con = AspectCritic(name="conciseness", llm=evaluator_llm, definition="Check if the answer is short but comprehensive.")
    metric_coh = AspectCritic(name="coherence", llm=evaluator_llm, definition="Check if the answer is consistent and logical.")
    metric_cor = AspectCritic(name="correctness", llm=evaluator_llm, definition="Check if the answer is relevant and accurate based on given documents.")
    
    #metric_noise = NoiseSensitivity(llm=evaluator_llm)
    #metric_context_recall = LLMContextRecall(llm=evaluator_llm)
    #metric_context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    response_text = generation(test_data2)  # Pass the list directly
    response = SingleTurnSample(user_input=test_data2[1]['content'], response=response_text)


    score_con = await metric_con.single_turn_ascore(response)  
    score_coh = await metric_coh.single_turn_ascore(response)  
    score_cor = await metric_cor.single_turn_ascore(response)  
    #score_noise = await metric_noise.single_turn_ascore(response)  
    #score_context_recall = await metric_context_recall.single_turn_ascore(response)
    #score_context_precision = await metric_context_precision.single_turn_ascore(response)

    print ("Conciseness Score:", score_con)
    print ("Coherence Score:", score_coh)
    print ("Correctness Score:", score_cor)
    #print ("Noise Sensitivity Score:", score_noise)
    #print ("Context Recall Score:", score_context_recall)
    #print ("Context Precision Score:", score_context_precision)

# Run the async function
asyncio.run(evaluate_test_data())




'''
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

test_data = {

    "user_input" : "What is warfighter function?",
    "response" : "A warfighting function  is a group of tasks and systems united by a common purpose that commanders use to accomplish mis sions and training objectives.  Warfighting functions are the physical means that tactical commanders use to execute operations and ac complish missions assigned by superior tactical- and operational-level commanders. The purpose of warf ighting functions is to provide an intellectual organization for common critical capabilities available to comm anders and staffs at all echelons and levels of warfare. "

}

test_data2 = [
        {
            "role": "system",
            "content": "You are a battalion executive and operations officer specialized in leading and conducting military decision-making process."
        },
        {
            "role": "user",
            "content": "I am an infantry battalion commander, and have been asked to secure a airstrip on Sicily Drop Zone, with a time on target of 0400 on 1 March, 2025.  My battalion, 2-508th PIR, will be augmented with an SAPPER engineer platoon and M777 artillary battery.  Draft a warning order, consistent with FM 5.0, in the 5 paragraph oporder format."
        }
    ]

metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Check if the answer is relevant and accurate based on given documents.")
test_data2 = generation(**test_data2)
metric.single_turn_ascore(test_data2)
'''


