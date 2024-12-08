from pydantic import BaseModel,Field
from langchain.tools import BaseTool
from typing import Type, TypedDict,Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage,HumanMessage,ToolMessage 
import operator
from langchain_google_genai import ChatGoogleGenerativeAI
#from vocal_order import Speech
from dotenv import load_dotenv
load_dotenv()

from track_object import track_specific_object_stream

yolo_class_names=['person',
'bicycle',
'car','motorcycle', 'airplane', 'bus', 'train', 'truck', 
'boat', 'traffic light', 'fire hydrant', 'stop sign', 
'parking meter', 'bench', 'bird', 'cat','dog', 'horse', 'sheep','cow','elephant', 
'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
'snowboard', 'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard', 
'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
'pizza','donut','cake', 'chair', 'couch', 'potted plant', 
 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
'refrigerator', 'book', 'clock','vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#order=Speech()


class AgentState(TypedDict):
  messages: Annotated[list[AnyMessage],operator.add]



class TrackInput(BaseModel):
    object_class: str = Field(description="should be a yolov8 class")



class CustomTrackTool(BaseTool):
  name:str="find_object"
  description:str=" This is a custom tool to find an object on video or frames sequence"
  args_schema:type[BaseModel]=TrackInput
  
  def _run(self,object_class:str):
      return track_specific_object_stream(object_class)
      #print("Oject tracked is around Manhattan at 7 P.M")
      #return "Oject tracked is around Manhattan at 7 P.M"

tool=CustomTrackTool()

def find_object():
  return "Object Tracked around Manhattan at 7 a.m !!!"

def describe_video():
  return "Frames Analyzed!!!"

tools={"find_object_fn":find_object,"describe_video_fn":describe_video}
class Agent:
  def __init__(self,model,tools,system=""):
    self.system=system 
    graph=StateGraph(AgentState)
    graph.add_node("llm",self.call_gemini)
    graph.add_node("action",self.take_action)
    graph.add_conditional_edges(
        "llm",
        self.exists_action,
        {True: "action",False: END}
    )
    graph.add_edge("action","llm")
    graph.set_entry_point("llm")
    self.graph=graph.compile()
    self.tools={t.name : t for t in tools}
    self.model=model.bind_tools(tools)#Il ya un problème avec la variable tools ici 

  def exists_action(self,state:AgentState):
    result=state["messages"][-1]
    return len(result.tool_calls)>0 

  def call_gemini(self,state:AgentState):
    messages=state["messages"]
    if self.system:
      messages=[SystemMessage(content=self.system)] + messages 
    message=self.model.invoke(messages)
    return {"messages":[message]}

  def take_action(self,state:AgentState):
    tool_calls=state["messages"][-1].tool_calls
    results=[]
    for t in tool_calls:
      print(f"Calling: {t}")
      if not t['name'] in self.tools:
        print("\n .... bad tool name ....")
        result="bad tool name, retry"
      else:
        result=self.tools[t['name']].invoke(t['args'])
      results.append(ToolMessage(tool_call_id=t['id'],name=t['name'],content=str(result)))
      print("Back to the model!")
      return {"messages":results}
    

prompt=f"""Tu es un assistant de Tracking intelligent nommé TrackMindBot.Utilisez l'outil de suivi pour rechercher des informations.\  
Vous êtes autorisé à effectuer plusieurs recherches (soit simultanément, soit de manière séquentielle).\  
Ne recherchez des informations que lorsque vous êtes sûr de ce que vous voulez.\  
Si vous avez besoin de rechercher des informations avant de poser une question complémentaire, vous êtes autorisé à le faire !\ 
La classe demandée par l'utilisateur doit être l'une de celles-ci {yolo_class_names} sinon répond directement à l'utilisateur que la classe rentrée ne figure pas parmi les objets détectés par le modèle utilisée .
"""
model=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")#,google_api_key=api_key)
TrackMindBot=Agent(model,[tool],system=prompt)

messages =[HumanMessage(content="Trouve-moi une personne")]

#messages =[HumanMessage(content=order)]
result=TrackMindBot.graph.invoke({"messages":messages})
print(result['messages'][-1].content)


####Subsitute prompt:

#You are smart tracking assistant.Use the track tool to loop up information.\
#You are allowed to make multiple calls (either together or in sequence).\
#Only look up information when you are sure of what you want.\
#If you need to look up some information before asking a follow up question,you are allowed to do that!
