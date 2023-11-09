class PriorityQueue(object):
	def __init__(self):
		self.queue = []

	def __str__(self):
		return ' '.join([str(i) for i in self.queue])

	# for checking if the queue is empty
	def isEmpty(self):
		return len(self.queue) == 0

	# for inserting an element in the queue
	def push(self, data):
		self.queue.append(data)

	# for popping an element based on Priority
	def pop(self):
		try:
			max_val = 0
			for i in range(len(self.queue)):
				# the data is stored as (index, val_acc)
				if self.queue[i][1] > self.queue[max_val][1]:
					max_val = i
			item = self.queue[max_val]
			del self.queue[max_val]
			return item
		except IndexError:
			print()
			exit()

	def top(self):
		# same as top but just wont delete
		max_val = 0
		for i in range(len(self.queue)):
			# the data is stored as (index, val_acc)
			if self.queue[i][1] > self.queue[max_val][1]:
				max_val = i
		item = self.queue[max_val]
		return item



if __name__ == '__main__':
	myQueue = PriorityQueue()
