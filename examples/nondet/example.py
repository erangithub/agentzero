from agentzero.llms import EchoLLM
from agentzero.environment import Environment


_price_counter = 0


def get_price(ticker):
    global _price_counter
    _price_counter += 1
    return {"ticker": ticker, "price": 100 + _price_counter}


def main():
    global _price_counter
    _price_counter = 0

    env = Environment()

    det_get_price = env.nondet(get_price)

    result1 = det_get_price("AAPL")
    print(f"First call: {result1}")  # expecting 101

    env.register_nondet(get_price)
    result2 = env.get_price("AAPL")
    print(f"Second call: {result2}") # expecting 102

    env.rewind()

    result_a = det_get_price("AAPL") # expecting 101
    print(f"After rewind, first call again: {result_a}")

    result_b = env.get_price("AAPL") # expecting 102
    print(f"After rewind, second call again: {result_b}")



if __name__ == "__main__":
    main()