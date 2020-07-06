
def replace(s, s1, s2, skip):
  pos = 0
  while (1):
    i = s.find(s1, pos)
    if (i == -1):
      break
    if (i > 0 and s[i-1] == skip):
      pos = i + 1
      continue
    s = s[:i] + s2 + s[i+len(s1):]
    pos = i + len(s2)
  return s

def tokenize(s):

  lt = []
  if (s[0] == '"'):
    s = "`` " + s[1:]
  s = s.replace(" \"", "  `` ")
  s = s.replace("(\"", "( `` ")
  s = s.replace("[\"", "[ `` ")
  s = s.replace("{\"", "{ `` ")
  s = s.replace("<\"", "< `` ")

  s = s.replace("...", " ... ")

  s = s.replace(",", " , ")
  s = s.replace(";", " ; ")
  s = s.replace(":", " : ")
  s = s.replace("@", " @ ")
  s = s.replace("#", " # ")
  s = s.replace("$", " $ ")
  s = s.replace("%", " % ")
  s = s.replace("&", " & ")

  pos = len(s) - 1;
  while (pos > 0 and s[pos] == ' '):
    pos = pos-1
  while (pos > 0):
    c = s[pos]
    if (c == '[' or c == ']' or c == ')' or c == '}' or c == '>' or
        c == '"' or c == '\''):
        pos-=1
        continue
    break
  if (pos >= 0 and s[pos] == '.' and not (pos > 0 and s[pos-1] == '.')):
    s = s[:pos] + " ." + s[pos+1:]
  
  s = s.replace("?", " ? ")
  s = s.replace("!", " ! ")
    
  s = s.replace("[", " [ ")
  s = s.replace("]", " ] ")
  s = s.replace("(", " ( ")
  s = s.replace(")", " ) ")
  s = s.replace("{", " { ")
  s = s.replace("}", " } ")
  s = s.replace("<", " < ")
  s = s.replace(">", " > ")

  s = s.replace("--", " -- ")

  s = " " + s
  s = s + " "
  
  s = s.replace("\"", " '' ")

  s = replace(s, "' ", " ' ", '\'');
  s = s.replace("'s ", " 's ")
  s = s.replace("'S ", " 'S ")
  s = s.replace("'m ", " 'm ")
  s = s.replace("'M ", " 'M ")
  s = s.replace("'d ", " 'd ")
  s = s.replace("'D ", " 'D ")
  s = s.replace("'ll ", " 'll ")
  s = s.replace("'re ", " 're ")
  s = s.replace("'ve ", " 've ")
  s = s.replace("n't ", " n't ")
  s = s.replace("'LL ", " 'LL ")
  s = s.replace("'RE ", " 'RE ")
  s = s.replace("'VE ", " 'VE ")
  s = s.replace("N'T ", " N'T ")

  s = s.replace(" Cannot ", " Can not ")
  s = s.replace(" cannot ", " can not ")
  s = s.replace(" D'ye ", " D' ye ")
  s = s.replace(" d'ye ", " d' ye ")
  s = s.replace(" Gimme ", " Gim me ")
  s = s.replace(" gimme ", " gim me ")
  s = s.replace(" Gonna ", " Gon na ")
  s = s.replace(" gonna ", " gon na ")
  s = s.replace(" Gotta ", " Got ta ")
  s = s.replace(" gotta ", " got ta ")
  s = s.replace(" Lemme ", " Lem me ")
  s = s.replace(" lemme ", " lem me ")
  s = s.replace(" More'n ", " More 'n ")
  s = s.replace(" more'n ", " more 'n ")
  s = s.replace("'Tis ", " 'T is ")
  s = s.replace("'tis ", " 't is ")
  s = s.replace("'Twas ", " 'T was ")
  s = s.replace("'twas ", " 't was ")
  s = s.replace(" Wanna ", " Wan na ")
  s = s.replace(" wanna ", " wanna ")

  lt = s.strip().split()
  return lt