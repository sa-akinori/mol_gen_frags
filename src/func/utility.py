import numpy as np
import pandas as pd
import re
import operator as op
from functools import reduce
from itertools import product
from inspect import stack
import multiprocessing 
import os
import io
import datetime
import time
import sys
import pickle
import random
from joblib import Parallel, delayed, cpu_count
import logging
import torch
import transformers
from rdkit import Chem

BASEPATH = os.path.dirname(__file__).replace('/src/func', '')

canonical_smiles = lambda smi: Chem.MolToSmiles(Chem.MolFromSmiles(smi))

def set_seed(seed=42):
    """
    Seed setting for reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True) 
    
    transformers.set_seed(seed)
    
def Sprint(*args, **kwargs):
	output = io.StringIO()
	print(*args, file=output, **kwargs)
	strs   = output.getvalue()
	output.close()
	return strs

def Print(*args, **kwargs):
	# adding the preline and post lines for the sentences
	line = '-'*20
	nargs = (line,) + args + (line,)
	print(*nargs, **kwargs)

	
def ParallelApplySeries(pd_series: pd.Series, func, njobs=-1):
	"""
	Wrapper function for applying parallel computation by apply function for a series 
	"""
	def apply_func(one_pd, func):
		return one_pd.apply(func)

	njobs = njobs if njobs != -1 else cpu_count() -1
	batch = np.array_split(pd_series, njobs)
	print(f'njobs: {njobs}')
	ret_list = Parallel(n_jobs=njobs)([delayed(apply_func)(one_series, func) for one_series in batch])
	return pd.concat(ret_list)
		

def np_nan(s):
	arr = np.empty(s)
	arr[:] = np.nan
	return arr

def flatten(t):
	# making a list flatten
	return [item for sublist in t for item in sublist]

def GetRNGs(n: int, seed: int):
	# fuctory class for random number genrators (RNG)
	rngs = []
	for i in range(n):
		rngs.append(np.random.RandomState(seed+i*100))
	return rngs


def SplitFileToFiles(infname: str, includeheader: bool, nlines_per_file: int, change_input_file_suffix: bool=True, keepinputfile: bool=True):
	# scan the file to count the number of lines
	with open(infname) as fp: 
		if includeheader:
			header = fp.readline()
		
		lines = [line for line in fp] # not sufficient when insufficient memory 
		if len(lines) < nlines_per_file: 
			print('no need to split. Skip the process')
			return 

		# split lines into nlines_per_file
		basename, extension = infname.rsplit('.',1)
		for idx, subset in enumerate(partitionIntoSubsets(lines, nlines_per_file)):
			outfname = f'{basename}_sub{idx}.{extension}'
			with open(outfname, 'w') as of:
				if includeheader:
					of.write(header)
				of.writelines(subset)

	if change_input_file_suffix and keepinputfile:
		os.rename(infname, f'{infname}.backup')    

	if not keepinputfile:
		os.remove(infname)    

def partitionIntoSubsets(data,sz):
	remainder = data
	while remainder:
		subset = remainder[:sz]
		remainder = remainder[sz:]
		yield subset

def DictSplit(in_dict, n_split):
	key_list = np.array_split(list(in_dict.keys()), n_split)
	ret_list = []
	for keys in key_list: 
		one_dict = {key:in_dict[key] for key in keys}
		ret_list.append(one_dict)

	return ret_list

def CatFiles(in_flist, out_file, remove_input=False):
	# Simple concatination
	with open(out_file, 'w') as out:
		for in_f in in_flist:
			with open(in_f, 'r') as infp:
				for line in infp:
					out.write(line)
	
	if remove_input:
		for in_f in in_flist:
			os.remove(in_f)


def RemoveWordFromStr(in_str, word, regular_express=True):
	if not regular_express:
		if word not in in_str:
			return in_str
	
		return ''.join(in_str.split(word))
	else:
		return re.sub(word, '', in_str)

def IsMac():
	# current computer is mac?
	if GetOS() == 'MAC':
		return True 
	else:
		return False

def GetOS():
	if sys.platform =='darwin':
		return 'MAC'
	if sys.platform =='linux':
		return 'Linux'
	if sys.platform == 'win32':
		return 'WINDOWS'
	else:
		print('{} is unkonwn to me'.format(sys.platform))
		return None

def AssertTerminate(equation, msg=None):
	if not equation:
		print(msg)
		sys.exit(1)

def areinstance(arr, t):
	if isinstance(arr, dict):
		for a in arr:
			if not isinstance(arr[a], t):
				return False
	else:
		for a in arr:
			if not isinstance(a, t):
				return False

	return True # passing the check

	
def cat(a, b, appender='_'):
	return a + appender + b

def GetTime(return_second=False, return_date=False):
	t = datetime.datetime.fromtimestamp(time.time())
	if return_second:
		return '{}{}{}_{}{}{}'.format(t.year, t.month, t.day, t.hour, t.minute, t.second)
	elif return_date:
		return '{}{}{}'.format(t.year,t.month,t.day)
	else:
		return '{}{}{}_{}{}'.format(t.year, t.month, t.day, t.hour, t.minute)
	

def numpy2DF(x, ref_pd=None, ref_idx=None, ref_col=None):
	if isinstance(ref_pd, pd.DataFrame) and (ref_idx is None) and (ref_col is None):
		return pd.DataFrame(x, index=ref_pd.index, columns=ref_pd.columns)
	elif ref_pd is None:
		return pd.DataFrame(x, index=ref_idx, columns=ref_col)
	else:
		return x

def MakeLogFP(fname, add_timestamp=False, allow_override=False):
	if add_timestamp:
		fname += '_{}.log'.format(GetTime())
	if not allow_override:
		fname = MakeFile(fname) # unique name    
	fp = open(fname, "w", 1) # line bufffering
	return fp

def WriteMsgLogStdout(fp=None, msg='',  supress_std_out=False, add_newline=True):
	if add_newline:
		msg +='\n'
	if not supress_std_out:
		print(msg)
	
	if hasattr(fp, 'read'): # file object testing 
		if not isinstance(msg, str): # converting to string
			msg = str(msg)
		fp.write(msg)
	

class LogFile():
	def __init__(self, fname, add_timestamp=False):
		if add_timestamp:
			current_time = '_{}.'.format(GetTime())
			fname = current_time.join(fname.rsplit('.',1))
		
		self.fp = open(fname, "w", 1) # 1 means without line buffering
		self.fname = fname

	def __enter__(self):
		return self
	
	def __exit__(self, exec_type, exc_value, traceback):
		# for with statement in python
		self.fp.close()

	def __call__(self, msg):
		self.write(msg)

	def write(self, msg, suppress_std_out=False, add_newline=True):
		if msg is None:
			return 
		if add_newline:
			msg +='\n'
		
		if not suppress_std_out: 
			print(msg)
		
		if not isinstance(msg, str): # converting to string
			msg = str(msg)
		self.fp.write(msg)

 
def GetRootLogger(file_name, 
				  showstd: bool=True, 
				  simple_format: bool=True, 
				  clearlog: bool=False,
				  clearhandler: bool=True
				  ):
	"""
	Wrapper function for setting logger (memorize utility information)
	"""
	logger = logging.getLogger() # get the root loger
	if clearhandler: 
		while logger.hasHandlers():
			logger.removeHandler(logger.handlers[0])
		
	logger.setLevel(logging.INFO)
	mode = 'w' if clearlog else 'a'
	fh = logging.FileHandler(file_name, mode=mode)
	fh.setLevel(logging.INFO)
	if not simple_format:
		f ='[%(levelname)-8s] [pid:%(process)d] [%(name)s]:[%(lineno)03d]:[%(message)s]'
	else:
		f = '%(asctime)s\t%(name)s\t%(message)s'
	fmt = logging.Formatter(f)
	fh.setFormatter(fmt)
	logger.addHandler(fh)
	if showstd:
		hd = logging.StreamHandler(sys.stdout)
		hd.setLevel(logging.INFO)
		logger.addHandler(hd)
		
	logger.info('Start logging')
	return logger



def makeDataFrameFromRef(x, ref_pd):
	"""
	make a dataframe frome reference dataframe
	"""
	if isinstance(ref_pd, pd.DataFrame):
		return pd.DataFrame(x, index=ref_pd.index, columns=ref_pd.columns)
	elif isinstance(ref_pd, pd.Series):
		return pd.Series(x.ravel(), index=ref_pd.index, name=ref_pd.name)
	else:
		return x


def pickUpStratifiedSamples(df, nbins, name='pot.(log,Ki)', r_seed=0):
	"""
	Pick up a representative (random) per bin (bin is the order)
	"""
	df_sort = df.sort_values(by=name, ascending=False)
	nsample = df.shape[0]
	bin_size = np.repeat(nsample/nbins, nbins)
	rem  = nsample % nbins
	
	# add 1 to the first rem bins in the bin_size
	for i in range(rem):
		bin_size[i] += 1
	
	# sampling based on the size
	strt_idx = 0
	selected = pd.DataFrame()
	prng = np.random.RandomState(r_seed)
	for i in range(nbins):
		end_idx = strt_idx + bin_size[i]
		select_idx = prng.choice(range(strt_idx, end_idx), size=1)[0]
		selected   = selected.append(df_sort.iloc[select_idx, :])
		strt_idx = end_idx

	return selected

def split_columns(x, pivot, axis=0):
	"""
	split rows (columns) into two sets based on the index of the columns
	:param x: matrix
	:return: x1, x2: x1 does not contain the pivot
	"""
	if axis==0:
		x1 = x[:pivot,:]
		x2 = x[pivot:,:]
	else:
		x1 = x[:,:pivot]
		x2 = x[:,pivot:]

	return x1, x2

def splitDFna_nonna(df):
	nan_df = df[df.isnull().any(axis=1)]
	ok_df = df.loc[~df.index.isin(nan_df.index)]
	return ok_df, nan_df

def RemoveKey(d, key):
	# remove contents in the key
	new_d = dict(d)
	del new_d[key]
	return new_d

class RegrexDict(dict):
	def get_matching(self, event):
		return(self[key] for key in self if re.match(key, event))

class logger():
	def __init__(self):
		self.xs = {}
		self.models = {}
		self.comments = {}


def isInRange(x, low, upp):
	if low < upp:
		return ( x >= low) & (x<= upp)
	else:
		return (x <= low) & (x >= upp)

def isInRangeRow(x_vec, low_vec, upp_vec):
	if isinstance(x_vec, pd.DataFrame):
		xvec = x_vec.as_matrix()
	
	if isinstance(low_vec, pd.Series):
		upps = upp_vec.as_matrix()
		lows = low_vec.as_matrix()
	
	if lows[0] > upps[0]:
		tmp = lows
		lows = upps
		upps = tmp
	
	is_in = []
	for i in xrange(0, xvec.shape[0]):
		is_in.append( np.sum(( (xvec[i,:] - lows) >= 0) & ((xvec[i,:] - upps) <= 0) ) == xvec.shape[1])
	
	return is_in

def argmax_val_range(func, x1, x2, pred=1.0E-03):
	x_range = np.arange(x1, x2, pred)
	idx_max = np.argmax(func(x_range.reshape(-1,1)))
	return x_range[idx_max]


def replace_strings(org_str, conditions):
	"""
	Replace multiple strings in a string

	input:
	------
	x_str:      string to be replaced
	conditions: replace conditions (dict) {'condition1':'change1', ...}
	
	output:
	-------
	text after replacement
	"""
	
	conditions = dict((re.escape(k), v) for k, v in conditions.iteritems())
	pattern    = re.compile("|".join(conditions.keys()))
	return   pattern.sub(lambda x: conditions[re.escape(x.group(0))], org_str)

def MakeRandomTuplesFromArray(iterable, r, m=1, buff=1000):
	# buff is to reduce the number of unique combinations
	pool    = tuple(iterable)
	n       = len(pool)
	sampled=[]
	for i in range(m+buff):
		sampled.append(tuple(pool[i] for i in sorted(random.sample(range(n), r))))
	return list(set(sampled))[:m]

def MakeRandomTuplesFromArrays(iters, m=1, buff=1000):
	# buff is to reduce the number of unique combinations
	pools   = [tuple(iterable) for iterable in iters]
	ns      = [len(pool) for pool in pools]
	sampled=[]
	for i in range(m+buff):
		sampled.append(tuple([pool[random.sample(range(i),1)[0]] for i, pool in zip(ns, pools)]))
	return list(set(sampled))[:m]


def MakeRandomCombinations(x, k, n, rng=None):
	"""
	Make a set of combinations of variables with k combination
	
	input:
	------
	rng:    np.random random generator
	x:      list from which k combinations are selected 
	k:      the number of combinations sampled 
	n:      the number of repetetion
	"""
	if rng is None:
		rng = np.random.RandomState(seed=0)
	
	nx = len(x)
	nr = nCr(nx, k) # possible combinations
	n_sample = min(n, nr) 

	ret_combinations = set()
	
	while len(ret_combinations) != n_sample:
		candi = tuple(np.sort(rng.choice(x, k, replace=False)))
		ret_combinations.add(candi)

	return ret_combinations

def nCr(n, r):
	"""
	Fast calculation of nCr
	"""
	r = min(r, n-r)
	numer = reduce(op.mul, range(n, n-r, -1), 1)
	denom = reduce(op.mul, range(1, r+1), 1)
	return numer//denom


def applyfunc_with_batch(func, x, batchsize=10000):
	"""
	Apply function to splitted x into reasonable number of data.
	This function should be called when using SVM or SVR sklearn.
	"""
	if isinstance(x, pd.DataFrame):
		x = x.values

	if len(x) > batchsize: # for reasonable prediction with many data
		start = 0
		y = np.zeros(len(x))
		for end in range(batchsize, len(x), batchsize):
			tx = x[start:end, :]
			ty = func(tx)
			y[start:end] = ty
			start = end
				
		# residues
		tx = x[start:, :]
		ty = func(tx)
		y[start:] = ty
	else:
		y = func(x)
	return y

def applyfunc_with_batch_mt(func, x, nworkers=8):
	"""
	Apply function to splitted x into reasonable number of data.
	This function should be called when using SVM or SVR sklearn.
	"""
	pass

	# if isinstance(x, pd.DataFrame):
	#     x = x.values

	# if nworkers == -1:
	#     nworkers = psutil.cpu_count() -1
	# split_x = np.array_split(x, nworkers)

	# pool = multiprocessing.Pool(nworkers)
	# ret_y = joblib.Parallel(n_jobs=nworkers)(joblib.delayed(func)(x_p) for x_p in split_x)
	# ret_y = np.concatenate(ret_y)
	# return ret_y

def search_exist_suffix(f_path):
	dirname, basename = os.path.split(f_path)

	for i in range(1,1000):
		if '.' in basename: # file
			n_name = basename.replace('.', '_{}.'.format(i))
		else:
			n_name = basename + '_' + str(i) # folder
		
		new_name = os.path.join(dirname,n_name)
		if not os.path.exists(new_name):
			return new_name
		# could not find unused filename for 1000 loops
	raise ValueError('cannot find unused folder names')

def MakeFile(file_path):
	"""
	Make a file at the path place (without overwrite)
	"""
	if os.path.exists(file_path):
		file_path = search_exist_suffix(file_path)
	return  file_path


def MakeFolderWithCurrentFuncName(folder_path, allow_override=False, skip_create=False, time_stamp=False):
	caller_name = str(stack()[1].function)
	outfname = f'{folder_path}/{caller_name}'
	return MakeFolder(outfname, allow_override=allow_override, skip_create=skip_create, time_stamp=time_stamp)


def MakeFolder(folder_path, allow_override=False, skip_create=False, time_stamp=False):
	"""
	Make a folder
	"""
	today = GetTime(return_date=True)
	if os.path.exists(folder_path) and skip_create:
		if time_stamp:
			return f'{folder_path}_{today}'
		return folder_path

	if os.path.exists(folder_path) and (not allow_override):
		Warning('Specified folder already exists. Create new one')
		folder_path = search_exist_suffix(folder_path)
		
	if time_stamp:
		folder_path = f'{folder_path}_{today}'
		
	os.makedirs(folder_path, exist_ok=allow_override)

	return folder_path
	
def MakeFolders(base_folder, list_parent_dirs, list_sub_dirs=None, create_base_fd=False, allow_override=False, return_parents=False):
	"""
	Make sub directories for conducting experiments.
	"""
	if create_base_fd:
		base_folder=MakeFolder(base_folder, allow_override)
	elif not os.path.exists(base_folder):
		ValueError('%s cannot exists.'%base_folder)
	
	parent_paths = dict()
	for p_dir_name in list_parent_dirs:
		path_dir = os.path.join(base_folder, p_dir_name)
		if os.path.exists(path_dir):
				print('%s already exists. Skip making the folder.'%p_dir_name)
		else:	
			os.mkdir(path_dir)
		
		if list_sub_dirs is not None:
			for dir_name in list_sub_dirs:
				sub_path_dir = os.path.join(path_dir, dir_name)
				if os.path.exists(sub_path_dir):
					print('%s already exists. Skip making the folder'%dir_name)
				else:
					os.mkdir(sub_path_dir)
		parent_paths[p_dir_name] = path_dir

	if return_parents:
		return parent_paths
	else: # base folder paths
		return base_folder
	
def find_substrs(name, black_list):
	"""
	True, iff name contains any of the words in the black_list
	"""
	for word in black_list:
		if word in name:
			return True
	return False # not found

def IsWordContained(org_str, word):
	return True if word in org_str else False 

def MergeColumns(table, col1, col2, use_col1_mask):
	"""
	Merge 2 columns into 1 column based on the use_col1_mask
	"""
	col1_idx = table.index[use_col1_mask]
	col2_idx = table.index[~use_col1_mask]

	return pd.concat([table.loc[col1_idx][col1], table.loc[col2_idx][col2]])

def ProductDict(**kwargs):
	keys = kwargs.keys()
	vals = kwargs.values()

	for instance in product(*vals): # positional argments
		yield dict(zip(keys, instance))

def IsDataFrame(matrix):
	if isinstance(matrix, pd.DataFrame):
		return True
	else:
		return False

def FindRows(query, matrix, return_index=False):
	# selecting rows matching the query
	if IsDataFrame(matrix):
		matrix = matrix.values
		query = query.values
	
	mask_match =  np.all(matrix == query, axis=1)
	
	if IsDataFrame(matrix):
		if return_index:
			return matrix[mask_match].index
		else:
			return matrix[mask_match]
	else:
		if return_index:
			return np.where(mask_match)[0]
		else:
			return matrix[mask_match, :]


def CatSeriesToDF(df, s):
	s_data = np.tile(s.values, (len(df),1))
	s_df = pd.DataFrame(s_data, index=df.index, columns=s.index)
	return pd.concat([df, s_df], axis=1)

def pickle_load(
	path:str
	):
	with open(path, 'rb') as f:
		data = pickle.load(f)
		
	return data
	
def pickle_save(
	path:str, 
	data,
	protocol=5
):
	with open(path, 'wb') as f:
		pickle.dump(data, f, protocol=protocol)
        
if __name__ =='__main__':
	a = MakeRandomTuplesFromArrays([range(1000), range(2000)], 100000, buff=1000)
	print(1)