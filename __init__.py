# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
WEB_DIRECTORY = "./web"

from .nodes import *
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PreViewVideo":PreViewVideo,
    "PromptNode":PromptNode,
    "LoraPathLoader":LoraPathLoader,
    "MotionLoraLoader":MotionLoraLoader,
    "I2V_AdapterNode": I2V_AdapterNode
}