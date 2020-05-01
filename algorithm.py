import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import os
import argparse

RESULTS_FILENAME = 'results.bin'

#these functions use dynamic programming for speedup

_F_cache = {}

def F(v):
	if not v in _F_cache:
		_F_cache[v] = (v**4)/4 - 2 * (v**3) - (11/2) * (v**2) - 5 * v

	return _F_cache[v] 

_F2_cache = {}

def F1(v):
	if not v in _F2_cache:
		_F2_cache[v] = v**3 - 6 * (v**2) - 11 * v - 5

	return _F2_cache[v]

_F3_cache = {}

def F2(v):
	if not v in _F3_cache:
		_F3_cache[v] = 3 * (v**2) - 12 * v - 11

	return _F3_cache[v]

#unstable interval (obtained theoretically)
unstable_v0 = -0.7589
unstable_v1 = 4.7689

#struct to put together the function and its derivatives
class Function:
	def __init__(self, x):
		self.f = F(x)
		self.first = F1(x)
		self.second = F2(x)

def algorithm(interval_0, interval_1, step, show_progress=False, save=True):
	best_diff = None

	best_v0 = None
	best_v1 = None

	best_m = None
	best_n = None

	v0 = interval_0[0]

	while interval_0[0] <= v0 <= interval_0[1]:
		f0 = Function(v0)
		m0 = f0.first
		n0 = f0.f - m0 * v0

		v1 = interval_1[0]

		if show_progress:
			#show progress as percentage
			print((interval_0[0] - v0)/(interval_0[0] - interval_0[1]) * 100)

		while interval_1[0] <= v1 <= interval_1[1]:
			f1 = Function(v1)
			m1 = f1.first
			n1 = f1.f - m1 * v1

			diff = abs(m0 - m1) + abs(n0 - n1)

			if not best_diff or diff < best_diff:
				best_diff = diff
				best_v0 = v0
				best_v1 = v1
				best_m = (m0 + m1)/2
				best_n = (n0 + n1)/2

			v1 += step

		v0 += step

	print('Best diff: %f' % best_diff)

	if save:
		#save the results to a file
		with open(RESULTS_FILENAME, 'wb') as f:
			pickle.dump(best_v0, f)
			pickle.dump(best_v1, f)
			pickle.dump(best_m, f)
			pickle.dump(best_n, f)

	return best_v0, best_v1, best_m, best_n

def draw(v_l, v_g, m, n, num):
	limits = (-5, v_l, unstable_v0, unstable_v1, v_g, 12)

	num /= len(limits)

	#these arrays hold the value of x, F and y = mx + n for every interval
	x = [None] * (len(limits) - 1)
	y_f = [None] * (len(limits) - 1)
	y_line = [None] * (len(limits) - 1)

	for i in range(len(limits) - 1):
		x[i] = np.linspace(limits[i], limits[i+1], num)
		y_f[i] = [F(v) for v in x[i]]
		y_line[i] = m * x[i] + n


	fig, ax = plt.subplots(1, 1)
	ax.set_xlim([limits[0], limits[-1]])

	#blue interval is drawn on top

	#points until v_l
	ax.plot(x[0], y_line[0], color='black')
	ax.plot(x[0], y_f[0], color='blue')

	#meta-estable points
	ax.plot(x[1], y_f[1], color='darkorange')
	ax.plot(x[1], y_line[1], color='blue')

	#unstable points
	ax.plot(x[2], y_f[2], color='red')
	ax.plot(x[2], y_line[2], color='blue')

	#meta-estable points
	ax.plot(x[3], y_f[3], color='darkorange')
	ax.plot(x[3], y_line[3], color='blue')

	#points after v_g
	ax.plot(x[4], y_line[4], color='black')
	ax.plot(x[4], y_f[4], color='blue')

	#markers
	marker = '.'
	ax.plot(v_l, F(v_l), color='blue', marker=marker)
	ax.plot(v_g, F(v_g), color='blue', marker=marker)
	ax.plot(unstable_v0, F(unstable_v0), color='red', marker=marker)
	ax.plot(unstable_v1, F(unstable_v1), color='red', marker=marker)

	#labels
	ax.set_ylabel('F', rotation=0)
	ax.set_xlabel('v')

	#custom legend
	legend_custom_lines = [Line2D([0], [0], color='blue'), Line2D([0], [0], color='darkorange'), Line2D([0], [0], color='red')]
	ax.legend(legend_custom_lines, ["Equilibrio estable", "Estados metaestables", "Estados inestables"])

	plt.savefig('figure.pdf')
	plt.show()

def parse_interval(s):
	try:
		v0, v1 = map(int, s.split(','))
		return v0, v1
	except:
		raise argparse.ArgumentTypeError("Interval must be v0, v1")

if __name__=='__main__':
	parser = argparse.ArgumentParser(description="Algorithm to obtain v_l and v_g numerically using Maxwell's construction")

	parser.add_argument('--force', help='force the execution of the algorithm and overwrite the previous result', action='store_true')
	parser.add_argument('--accuracy', help='accuracy to which the values are calculated', type=float, default=0.00005)
	parser.add_argument('--show-progress', help='show calculation progress as a percentage', action='store_true')
	parser.add_argument('--first-interval', help='select the interval (v0, v1) in which v_l is contained', type=float, nargs=2, default=[-3.5, -2.5], metavar=('v0', 'v1'))
	parser.add_argument('--second-interval', help='select the interval (v0, v1) in which v_g is contained', type=float, nargs=2, default=[6.5, 7.5], metavar=('v0', 'v1'))
	parser.add_argument('--dont-save', help='dont save the results to a file', action='store_true')

	args = parser.parse_args()

	if not args.force and os.path.isfile(RESULTS_FILENAME):
		#load the results from previous file
		with open(RESULTS_FILENAME, 'rb') as f:
			v_l = pickle.load(f)
			v_g = pickle.load(f)
			m = pickle.load(f)
			n = pickle.load(f)
	else:
		#calculate results and save them to file
		v_l, v_g, m, n = algorithm(args.first_interval, args.second_interval, step=args.accuracy, show_progress=args.show_progress, save=(not args.dont_save))

	print('v_l: %f' % v_l)
	print('v_g: %f' % v_g)
	print('m: %f' % m)
	print('n: %f' % n)

	draw(v_l, v_g, m, n, num=1000)
	