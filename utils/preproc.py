import re, html

inputfile='all.txt'

def rePrint(x):
	srch = re.compile(x)
	print(' '.join(sorted(list(set(sum([sum([srch.findall(p) for p in c], []) for c in convers], []))))))
	
	
def apply(x, y, s=[]):
	if s==[]: s=convers
	return [[re.sub(x, y, p) for p in c] for c in s]
	
	
lines = [l.strip().split('\t')[1:] for l in open(inputfile, encoding="utf-8").readlines()[1:]]
convers = [l[:3] for l in lines]
labels = [l[3] for l in lines]

convers = [[html.unescape(p).lower() for p in c] for c in convers]
convers = apply('([a-zA-Z]+)0([a-zA-Z]+)?', r'\1o\2')
convers = apply('[\"\`\’]+', "'")
convers = apply(r'[\/\\\=\+]+', '')
convers = apply('([a-z]+)[\']+([^a-z\']+)', r'\1 \2')
convers = apply('([^a-z\']+)[\']+([a-z]+)', r'\1 \2')
convers = apply('([^a-z\']+)[\']+([^a-z\']+)', r'\1 \2')
convers = apply('([^a-z\']+)[\']+([^a-z\']+)', r'\1 \2')
convers = apply('([a-z]+)\-([a-z]+)', r'\1\2')
convers = apply('([a-z]+)\-([a-z]+)', r'\1\2')
convers = apply('([\x00-\xFF])([^\x00-\xFF])', r'\1 \2')
convers = apply('([^\x00-\xFF])([\x00-\xFF])', r'\1 \2')
convers = apply(r'(.)\1+', r'\1\1')
convers = apply(r'([^a-z]{2})\1+', r' \1 ')
convers = apply(r'([\.\!\?])\1', r' \1\1 ')
convers = apply(r'([^\.])\.', r'\1 .')
convers = apply(r'\.([^\.])', r'. \1')
convers = apply(r'\. \.', ' .. ')
convers = apply(r'\d{2,}(?:\,\d+)?', ' ')
convers = apply(r'(?:\d+)?[^a-z\d \>\<\(\)]\d+', '')
convers = apply('([a-z]{2,})([^a-z\'])', r'\1 \2')
convers = apply('\s+', ' ')
convers = apply('[^a-z ]{8,}', '')

utfemots = apply('[\x00-\xFF]', '')
convers = apply('[^\x00-\xFF]', '')
ascemots = apply('[a-z]{2,}', ' ')
ascemots = apply(' [a-z] ', ' ', ascemots)
ascemots = apply('^[a-z] ', '', ascemots)
ascemots = apply(' [a-z]$', '', ascemots)
ascemots = apply('^\s*[a-z]\s*$', '', ascemots)
ascemots = apply('[a-z]?\'[a-z]?', '', ascemots)
ascemots = apply('\s+', ' ', ascemots)
utfemots = apply('(.)', ' \1 ', utfemots)
utfemots = apply('\s+', ' ', utfemots)
convers = apply('[^a-z ]{2,}', '')
convers = apply('\s[^a-z][a-z](\s|$)', ' ')
convers = apply('\s+', ' ')

#rePrint('[a-z]+\'[a-z]+')


open('utfemots.txt', 'w+', encoding="utf-8").write('\n'.join(['\t'.join(c) for c in utfemots]))
open('convers.txt', 'w+', encoding="utf-8").write('\n'.join(['\t'.join(c) for c in convers]))
open('ascemots.txt', 'w+', encoding="utf-8").write('\n'.join(['\t'.join(c) for c in ascemots]))

