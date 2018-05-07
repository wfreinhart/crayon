#
# parallel.py
# wrapper for easy scripting of MPI-parallelized tasks
#
# Copyright (c) 2018 Wesley Reinhart.
# This file is part of the crayon project, released under the Modified BSD License.

import sys
import math
import time

import numpy as np

try:
    from mpi4py import MPI
    parallel_enabled = True
except:
    print('crayon.parallel functionality requires mpi4py')
    parallel_enabled = False

def info():
    R""" get MPI info

    Returns:
        comm (MPI.COMM_WORLD): communicator used by MPI
        size (int): number of threads
        rank (int): thread identity
        master (bool): is this thread the master?
    """
    if not parallel_enabled:
        return None, 1, 0, True
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    master = ( rank == 0 )
    return comm, size, rank, master

def partition(l):
    R""" partition a list into roughly equal blocks for parallel operations

    Args:
        l (list): the list to partition

    Returns:
        r (list): the portion of the list to be used by the active thread
    """
    comm, size, rank, master = info()
    division = len(l) / float(size)
    p = [ l[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(size) ]
    q = [str(s) for s in p]
    r = [p[x] for x in np.argsort(q)]
    return r[::-1][rank]

class ETA:
    R""" handle reporting estimated time remaining during a task

    Args:
        n (int,optional): number of periods (default 0)
        reports (int,optional): number of reports between all periods (default 1)
    """
    def __init__(self,n=0,reports=1):
        # record start time
        self.startTime = time.time()
        # calculate reporting interval
        self.n = n
        self.interval = int(math.ceil(float(n)/reports))
        if self.interval < 1:
            self.interval = 1
    def report(self,i):
        R""" reports the elapsed time and estimated time remaining

        Args:
            i (int): the current period / time step / task number
        """
        if i > 0 and i%self.interval == 0:
            elapsed = time.time()-self.startTime
            estimate = elapsed * self.n / float(i)
            print('Completed %d of %d (%.0f percent) : %.1f s of %.1f s'%
                  (i,self.n,float(i)/self.n*100.,elapsed,estimate))
            sys.stdout.flush()

class ParallelTask:
    R""" convenient wrapper for coordinating parallel tasks using MPI4PY
    """
    def __init__(self):
        self.comm, self.size, self.rank, self.master = info()
        self.data = None
        self.solo_mode = False
        if self.master:
                print('ParallelTask initialized with %d MPI ranks'%self.size)
        if self.size < 2:
            print('Warning: ParallelTask is only aware of one MPI rank! Computing in solo mode.')
            self.solo_mode = True
    def shareData(self,data):
        R""" copy object from master thread to slave threads

        Args:
            data: any object (only master copy is used)

        Returns:
            data: copy of the object from master thread
        """
        if self.master:
            self.data = data
            for i in range(1,self.size):
                self.comm.send(data, dest=i, tag=0)
        else:
            sys.stdout.flush()
            self.data = self.comm.recv(source=0, tag=0)
        return self.data
    def gatherData(self,data):
        R""" collect data from slave threads on master thread

        Args:
            data: any object

        Returns:
            datas (list): data objects from slave threads in order
        """
        if self.master:
            datas = []
            for i in range(1,self.size):
                buff = self.comm.recv(source=i, tag=0)
                datas.append(buff)
            return datas
        else:
            self.comm.send(data, dest=0, tag=0)
            return None
    def computeQueue(self,function=None,tasks=None,reports=10):
        R""" computes a specified function using a master-slave work-sharing paradigm

        Args:
            function (function handle): lambda function to be called with (task,self.data)
            tasks (list): list of inputs to lambda function to be used one at a time
            reports (int,optional): number of times to report estimated time remaining (default 10)

        Returns:
            results (list): result from each task in same order
        """
        if self.solo_mode:
            return self.soloCompute(function,tasks,reports)
        if self.master:
            return self.masterCompute(tasks,reports)
        else:
            return self.slaveCompute(function)
    def slaveCompute(self,function):
        R""" compute a function on a slave thread, waiting for master to send instructions
             and receive results

        Args:
            function (function handle): lambda function to be called with (task,self.data)
        """
        while True:
            # await instructions
            (index, task) = self.comm.recv(source=0, tag=1)
            if task is None:
                break
            # perform the task, supplying data synced across ranks
            result = function(task,self.data)
            # send back to master
            buff = (self.rank, index, task, result)
            self.comm.send(buff, dest=0, tag=2)
        return None
    def masterCompute(self,tasks,reports):
        R""" handle work assignment to slave threads

        Args:
            tasks (list): list of inputs to lambda function to be used one at a time
            reports (int): number of times to report estimated time remaining

        Returns:
            results (list): result from each task in same order
        """
        # set up accounting system for tasks
        nThreads = min(self.size,len(tasks))
        busyThreads = []
        idleThreads = [ x for x in range(1,nThreads) if x not in busyThreads ]
        sendCount = 0
        recvCount = 0
        countdown = ETA(len(tasks),reports)
        # release unecessary ranks right away
        for i in range(len(tasks),self.size):
            self.comm.send((None,None), dest=i, tag=1)
        # run through the queue
        results = [i for i in range(len(tasks))]
        while recvCount < len(tasks):
            while ( len(idleThreads) > 0 ) and ( sendCount < len(tasks) ):
                # send tasks to idle threads
                dest = idleThreads.pop()
                buff = (sendCount,tasks[sendCount])
                self.comm.send(buff, dest=dest, tag=1)
                busyThreads.append(dest)
                sendCount += 1
            # receive completed tasks
            buff = self.comm.recv(source=MPI.ANY_SOURCE, tag=2)
            rank, index, task, result = buff
            recvCount += 1
            countdown.report(recvCount)
            results[index] = result
            busyThreads.remove(rank)
            # release workers when tasks are done
            if sendCount == len(tasks):
                self.comm.send((None,None), dest=rank, tag=1)
            else:
                idleThreads.append(rank)
        # report elapsed time
        elapsed = time.time()-countdown.startTime
        print('Parallel queue complete. Execution time: ',elapsed)
        # return result
        return results
    def soloCompute(self,function,tasks,reports):
        R""" compute all tasks on master thread when no slave threads are available

        Args:
            function (function handle): lambda function to be called with (task,self.data)
            tasks (list): list of inputs to lambda function to be used one at a time
            reports (int,optional): number of times to report estimated time remaining (default 10)

        Returns:
            results (list): result from each task in same order
        """
        countdown = ETA(len(tasks),reports)
        results = []
        for i in range(len(tasks)):
            results.append( function(tasks[i], self.data ) )
            countdown.report(i)
        # report elapsed time
        elapsed = time.time()-countdown.startTime
        print('Solo queue complete. Execution time: ',elapsed)
        # return result
        return results
