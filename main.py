from modules import *

import warnings
warnings.filterwarnings('ignore')


        
if __name__ == "__main__":

    eps = 1e-32
    n_epoch = 200
    es_thres = 10

    fee = 0.0005 # transaction fee
    ws = 14*6
    dropout = 0.3

    data = pd.read_csv('data.csv') # load data
    N = data.shape[1]
    timepoints = ['202'] # timepoint for data split
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Train
    # 1. Set Hyperparameters
    n_layer, hidden_dim, d_model, n_head, batch_size, lr, k_clust = 2, 16, 8, 4, 32, 0.001, 5
    
    _params = dict()
    _params['dropout'] = dropout
    _params['n_layer'] = n_layer

    _params['n_head'] = n_head
    _params['d_model'] = d_model
    _params['hidden_dim'] = hidden_dim


    fname_pt = 'es.pt' # filename for saving model - early stopping

    # 2. Get dataset
    train_dataset, valid_dataset, test_dataset = get_dataset(data, timepoints, ws, device)        

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize model
    model = pt_model(_params, N, ws, k_clust, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # 4. Start learning
    es_last = -1
    loss_min = 1e+32
    temp = 0.1
        
    for epoch in range(n_epoch) :

        model.train()            
        for r_ws, r in train_loader :
            (z, z_prob), (c, s, r_hat, gamma), w = model(r_ws, temp=temp)            
            loss = custom_loss(w, r, z_prob, gamma, fee=fee)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        temp = temp * 2**(1/(np.log(epoch+2)**2))
        model.eval()
        torch.cuda.empty_cache()
        _, (sr_valid, cumrtn_valid) = get_measures(model, valid_loader, fee=fee, opt_return_all=False)
                
        if (loss_min > -sr_valid ) :
            loss_min = -sr_valid
            torch.save(model.state_dict(), fname_pt)
            es_last = epoch
        
        if epoch - es_last > es_thres :
            break



    ### Test Results
    model.load_state_dict(torch.load(fname_pt))
    model.eval()

    _, (sr_test_0, cumrtn_test_0) = get_measures(model, test_loader, fee=0, opt_return_all=False)

    _, (sr_test, cumrtn_test) = get_measures(model, test_loader, fee=fee, opt_return_all=False)
    
    print(f'Result(Without fees): cum. returns of {round(cumrtn_test_0, 3)} and sharpe ratio of {round(sr_test_0, 3)}')
    print(f'Result(With fees): cum. returns of {round(cumrtn_test, 3)} and sharpe ratio of {round(sr_test, 3)}')