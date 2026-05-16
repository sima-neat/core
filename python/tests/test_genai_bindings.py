import pyneat


def test_genai_value_types_and_text_sample_helpers():
  req = pyneat.GenerationRequest()
  req.prompt = "hello"
  req.system_prompt = "be concise"
  req.max_new_tokens = 8
  req.temperature = 0.25
  req.top_p = 0.9
  req.messages = [pyneat.ChatMessage()]
  req.messages[0].role = "user"
  req.messages[0].content = "hello"

  assert req.prompt == "hello"
  assert req.system_prompt == "be concise"
  assert req.messages[0].content == "hello"

  tensor = pyneat.Tensor.from_text("token")
  assert tensor.to_text() == "token"

  sample = pyneat.make_text_sample("prompt", "What is NEAT?")
  assert sample.kind == pyneat.SampleKind.TensorSet
  assert sample.port_name == "prompt"
  assert sample.stream_label == "prompt"
  assert sample.to_text() == "What is NEAT?"

  result = pyneat.GenerationResult()
  result.text = "done"
  result.finish_reason = "stop"
  result.metrics.generated_tokens = 1
  assert result.metrics.generated_tokens == 1

  token = pyneat.TokenSample()
  token.text = "d"
  token.is_final = False
  assert token.text == "d"


def test_genai_top_level_and_namespace_aliases_exist():
  assert pyneat.genai.VisionLanguageModel is pyneat.VisionLanguageModel
  assert pyneat.genai.GenAIModel is pyneat.GenAIModel
  assert pyneat.genai.GenerationRequest is pyneat.GenerationRequest
  assert hasattr(pyneat.genai.nodes, "language")
  assert hasattr(pyneat.graph.nodes, "genai_language")

  options = pyneat.genai.nodes.LanguageOptions()
  options.system_prompt = "Answer exactly."
  options.max_new_tokens = 24
  options.streaming = False
  assert options.system_prompt == "Answer exactly."
  assert options.max_new_tokens == 24
  assert options.streaming is False
