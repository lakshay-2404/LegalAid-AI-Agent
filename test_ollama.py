from langchain_ollama.llms import OllamaLLM
import pprint

m = OllamaLLM(model="deepseek-r1")
res = m.generate(["Hello"])
print("TYPE:", type(res))
print("\nDIR:")
print([n for n in dir(res) if not n.startswith("_")][:200])
print("\nREPR (full):")
pprint.pprint(res)