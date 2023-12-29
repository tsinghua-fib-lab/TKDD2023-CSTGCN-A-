import os
import time
import numpy as np
import torch
import sys
import util_pre
from engine import trainer
from torch.utils.tensorboard import SummaryWriter
from earlystopping import EarlyStopping


class Logger(object):
    def __init__(self, filename: str, mode: str = "a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

basepath = "./"
dataset = "garage/"

save_path = basepath+dataset
if not os.path.exists(save_path):
    os.makedirs(save_path)
writer = SummaryWriter(save_path+"log")
if not os.path.exists(save_path+"model/"):
    os.makedirs(save_path+"model/")
early_stopping = EarlyStopping(save_path = save_path+"model/", patience=100)
def main():
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    adj_mx = util_pre.load_adj(basepath+"dat_gmodel/adj_mx_QT.txt")
    adj_mx = torch.tensor(adj_mx).to(device).float()
    dataloader = util_pre.load_dataset(basepath+'data_gmodel/pstm_QT',basepath+'data_gmodel/em_QT',32,16,32)
    scaler = dataloader['scaler']
    
    supports = None
    expid = 5
    sys.stdout = Logger(save_path + "exp_QT_" + str(expid)+"_output.txt", "a")
    engine = trainer(scaler, 4, 24, 1161, 32, 0.3, 0.003, 0.0001, device, supports,adj_mx)

    if os.path.exists(save_path+"model/best_network.pt"):
        engine.model.load_state_dict(torch.load(save_path+"model/best_network.pt", map_location='cpu'))
    

    print("start training...", flush = True)
    his_loss = []
    val_time = []
    train_time = []      
    for i in range(1, 1000+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y, events) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3).contiguous()
            events = torch.LongTensor(events).to(device)
            metrics = engine.train(trainx, trainy[:, 0, :, :], events)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        for iter, (x, y, events) in enumerate(dataloader['val_loader'].get_iterator()):
            valx = torch.Tensor(x).to(device)
            valx = valx.transpose(1, 3)
            valy = torch.Tensor(y).to(device)
            valy = valy.transpose(1, 3).contiguous()
            events = torch.LongTensor(events).to(device)

            metrics = engine.eval(valx, valy[:, 0, :, :],events)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        
        test_loss = []
        test_mape = []
        test_rmse = []
        st1 = time.time()
        for iter, (x, y, events) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3).contiguous()
            events = torch.LongTensor(events).to(device)

            metrics = engine.eval(testx, testy[:, 0, :, :],events)
            test_loss.append(metrics[0])
            test_mape.append(metrics[1])
            test_rmse.append(metrics[2])

        mtest_loss = np.mean(test_loss)
        mtest_mape = np.mean(test_mape)
        mtest_rmse = np.mean(test_rmse)
        st2 = time.time()


        x = torch.Tensor(dataloader['x_test']).to(device)
        x = x.transpose(1, 3)
        x_event = torch.LongTensor(dataloader['x_test_event']).to(device)
        outs = []
        seg_embs = []
        for i in range(18):
            output, seg_emb = engine.model(x[i*16:(i+1)*16],x_event[i*16:(i+1)*16])
            output = output.transpose(1, 3)
            output = output.detach().cpu().numpy()
            seg_emb = seg_emb.detach().cpu().numpy()
            output = scaler.inverse_transform(output)
            outs.append(output)
            seg_embs.append(seg_emb)   
        data_time_pre = np.vstack(outs) 
        data_time_pre = np.absolute(data_time_pre).squeeze()
        MAPE_t = np.mean(np.abs(data_time_pre-dataloader["y_test"][...,0].transpose(0,2,1))/dataloader["y_test"][...,0].transpose(0,2,1))
        print("MAPE_TEST:",MAPE_t)


        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
        log = 'Epoch: {:03d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Validation Time: {:.4f}/epoch'
        print(log.format(i, mtest_loss, mtest_mape, mtest_rmse, (st2 - st1)), flush=True)
        writer.add_scalars("loss",{ "train_loss": mtrain_loss, "val_loss": mvalid_loss, "test_mae":np.mean(test_loss)}, i)
        writer.add_scalars("mape",{ "train_mape": mtrain_mape, "val_mape": mvalid_mape, "test_mape":np.mean(test_mape)}, i)
        writer.add_scalars("rmse",{ "train_rmse": mtrain_rmse, "val_rmse": mvalid_rmse, "test_rmse":np.mean(test_rmse)}, i)

        early_stopping(MAPE_t, engine.model)
        if (not os.path.exists(save_path+"TestTime_nasf.npy")) or (early_stopping.counter == 0):
            seg_embs = np.vstack(seg_embs)
            seg_embs = np.insert(seg_embs,0,np.zeros([1,24]),1)
            np.save(save_path+"/TestTime_pre.npy",data_time_pre.squeeze())
            np.save(save_path+"/TestSeg_emb.npy",seg_embs)
            
        if early_stopping.early_stop:
            break  

    print('Training finish!')
    print("MAPE_TEST:",MAPE_t)

if __name__ == "__main__":
    main()

