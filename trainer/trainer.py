import copy
import time

from tensorboardX import SummaryWriter

from utils import *


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, checkpoint_folder, device, num_epochs=25):
    """Start training"""
    start_time = time.time()
    writer = SummaryWriter()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    iter = 0
    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(str(epoch), str(num_epochs - 1)))
        logging.info(' ')
        for i,datapoint in enumerate(train_loader):
            datapoint['input_image']=datapoint['input_image'].type(torch.FloatTensor)
            datapoint['label']=datapoint['label'].type(torch.LongTensor)

            input_image = Variable(datapoint['input_image'].to(device))
            label = Variable(datapoint['label'].to(device))
            
            model.train()
            optimizer.zero_grad()
            outputs = model(input_image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Training_Loss',loss.item()/outputs.shape[0], iter)
            iter=iter+1
            if iter % 10 == 0 or iter==1:
                # Calculate Accuracy on Validation set
                val_accuracy, val_loss = get_accuracy_over_set(val_loader, model, criterion, device)
                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    best_model_wts = copy.deepcopy(model.state_dict())
                writer.add_scalar('Validation_Loss',val_loss, iter) 
                writer.add_scalar('Validation_Accuracy',val_accuracy, iter) 

                time_since_beg=(time.time()-start_time)/60
                logging.info('Iteration: {}. Training Loss: {}. Validation Loss: {}. Time(mins) {}'.format(iter, loss.item()/outputs.shape[0], val_loss, time_since_beg))
            
            if iter % 500 ==0:
                if not os.path.exists(checkpoint_folder):
                    os.mkdir(checkpoint_folder)
                    
                torch.save(model,checkpoint_folder+'model_iter_'+str(iter)+'.pt')
                logging.info("model saved at iteration : "+str(iter))
                
        scheduler.step()
        
    model.load_state_dict(best_model_wts)
    return model
