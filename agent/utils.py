import time
import torch
import os
import random
import numpy as np
from utils.logger import log_to_screen, log_to_tb_val
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger
import numpy 
import hvwfg

def cal_ps_hv(pf, pf_num, ref, ideal=None):
    if ideal is None and ref.shape[0] == 2:
        ideal = numpy.array([0, 0])
    elif ideal is None and ref.shape[0] == 3:
        ideal = numpy.array([0, 0, 0])
    batch_size = pf.shape[0]
    
    hvs = numpy.zeros([batch_size, 1])
    specific_hv = numpy.zeros([batch_size,1])
    ref_region = 1
    for i in range(ref.shape[0]):
        ref_region = ref_region * (ref[i] - ideal[i])
    for k in range(batch_size):
        num = pf_num[k]
        hv = hvwfg.wfg(pf[k][:num].astype(float), ref.astype(float))
        specific_hv[k] = hv
        hv = hv / ref_region
        hvs[k] = hv

    return specific_hv,hvs

def gather_tensor_and_concat(tensor):
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)

def validate(rank, problem, agent, val_dataset, tb_logger, distributed = False, _id = None):
            
    # Validate mode
    opts = agent.opts
    if rank==0: print(f'\nInference with x{opts.val_m} augments...', flush=True)
    
    agent.eval()
    problem.eval()
    
    if opts.eval_only:
        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)
    
    val_dataset = problem.make_dataset(size=opts.graph_size,
                               num_samples=opts.val_size,
                               filename = val_dataset,
                               DUMMY_RATE = opts.dummy_rate)

    if distributed and opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=opts.world_size, rank = rank)
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.actor = torch.nn.parallel.DistributedDataParallel(agent.actor, device_ids=[rank])
        
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))

        assert opts.val_size % opts.world_size == 0
        train_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size = opts.val_size // opts.world_size, shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=train_sampler)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=opts.val_size, shuffle=False,
                                   num_workers=0,
                                   pin_memory=True)
    
    s_time = time.time()
    
    n_sols = opts.n_sols
    prefs = torch.zeros(n_sols, 2).cuda()
    for i in range(n_sols):
        prefs[i, 0] = 1 - i / (n_sols - 1)
        prefs[i, 1] = i / (n_sols - 1)
        
        
        
    timer_start = time.time()
    
    for batch_id, batch in enumerate(val_dataloader):
        assert batch_id < 1
        
        sols = np.zeros([opts.val_size, n_sols, 2])
        
        for st in range(prefs.shape[0]):
            pref = prefs[st, :]
            print("pref:" )
            print(pref)
            
            bv, cost_hist, best_hist, r, rec_history = agent.rollout(problem,
                                                                    opts.val_m,
                                                                    batch,
                                                                    pref,
                                                                    do_sample = True,
                                                                    record = False,
                                                                    show_bar = rank==0)
        
            if distributed and opts.distributed:
                dist.barrier()
                initial_cost = gather_tensor_and_concat(cost_hist[:,0].contiguous())
                time_used = gather_tensor_and_concat(torch.tensor([time.time() - s_time]).cuda())
                bv = gather_tensor_and_concat(bv.contiguous())
                costs_history = gather_tensor_and_concat(cost_hist.contiguous())
                search_history = gather_tensor_and_concat(best_hist.contiguous())
                reward = gather_tensor_and_concat(r.contiguous())
                dist.barrier()
            else:
                initial_cost = cost_hist[:,0]
                time_used = torch.tensor([time.time() - s_time])
                bv = bv
                costs_history = cost_hist
                search_history = best_hist
                reward = r
                
            # log to screen  
            if rank == 0: log_to_screen(time_used, 
                                        initial_cost, 
                                        bv, 
                                        reward, 
                                        costs_history,
                                        search_history,
                                        batch_size = opts.val_size, 
                                        dataset_size = len(val_dataset), 
                                        T = opts.T_max)
            
            # log to tb
            if(not opts.no_tb) and rank == 0:
                log_to_tb_val(tb_logger,
                            time_used, 
                            initial_cost, 
                            bv, 
                            reward, 
                            costs_history,
                            search_history,
                            batch_size = opts.val_size,
                            val_size =  opts.val_size,
                            dataset_size = len(val_dataset), 
                            T = opts.T_max,
                            epoch = _id)
                
            if distributed and opts.distributed: dist.barrier()
    
        
        timer_end = time.time()
        total_time = timer_end - timer_start
        
        
        
        
        if problem.real_size == 20:
            ref = np.array([30,3])    #20
        elif problem.real_size == 50:
            ref = np.array([40,3])   #50
        elif problem.real_size == 100:
            ref = np.array([60,3])   #100
        
        p_sols_num = np.full((opts.val_size,), n_sols)
    
        specific_hv,hvs = cal_ps_hv(pf=sols, pf_num=p_sols_num, ref=ref)
    
        print("HV: {:.4f}".format(specific_hv.mean()))
        print('HV Ratio: {:.4f}'.format(hvs.mean()))
        
        print('Run Time(s): {:.4f}'.format(total_time))
        
        
    
    
     
       