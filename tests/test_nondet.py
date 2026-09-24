from agentzero import Session
from agentzero.llms import EchoLLM

_price_counter = 0


def get_price(ticker):
    global _price_counter
    _price_counter += 1
    return {"ticker": ticker, "price": 100 + _price_counter}


def test_nondet_replay():
    global _price_counter
    _price_counter = 0

    env = Session().root
    env.register_llm_fn(EchoLLM().complete)
    env.register_input_fn(lambda: "")
    det_get_price = env.nondet(get_price)

    result1 = det_get_price("AAPL")
    assert result1 == {"ticker": "AAPL", "price": 101}
    env.register_nondet(get_price)
    result2 = env.get_price("AAPL")

    env.rewind()
    result_a = det_get_price("AAPL")
    assert result_a == result1
    result_b = env.get_price("AAPL")
    assert result_b == result2

    assert result_a != result_b

    print("test_nondet_replay passed")


if __name__ == "__main__":
    test_nondet_replay()
