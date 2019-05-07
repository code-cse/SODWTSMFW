import execnet


def call_python_version(Version, Module, Function, ArgumentList):
    gw      = execnet.makegateway("popen//python=python%s" % Version)
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    return channel.receive()

def my_function(X, Y): 
    return "Hello %s %s!" % (X, Y)


result = call_python_version("2.7", "my_module", "my_function",  
                             ["Mr", "Bear"]) 
print(result) 
result = call_python_version("2.7", "my_module", "my_function",  
                             ["Mrs", "Wolf"]) 
print(result)