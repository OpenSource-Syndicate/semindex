class Greeter:
    def hello(self, name):
        return f'Hello, {name}!'


def greet(name):
    return Greeter().hello(name)
