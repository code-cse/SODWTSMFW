import subprocess

    



input = open('input.txt','r').read()
# command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-preserveLines', '-lowerCase']
command = ['java', '-cp', '../stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar', 'edu.stanford.nlp.process.PTBTokenizer', '-preserveLines']  #, '-lowerCase'
#article = subprocess.check_output(command)

ps = subprocess.Popen(('echo', input), stdout=subprocess.PIPE)
output = subprocess.check_output(command, stdin=ps.stdout)
ps.wait()
print output
