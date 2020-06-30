# some housekeeping libraries
from absl import app, flags, logging
import sh

# import pytorch
import torch as th

# import pytorch_lightning
# this library simplifies training significantly
# familiarize yourself with the principals first: https://pytorch-lightning.readthedocs.io/en/latest/
import pytorch_lightning as pl

# hugging face libraries
# https://github.com/huggingface/nlp/
import nlp
# https://github.com/huggingface/transformers/
import transformers

# parameters for the network
# these have been tested and the network trains appropriately when implemented correctly
# use FLAGS.debug=True to test your network (it will not run an entire training epoch for this) see: https://pytorch-lightning.readthedocs.io/en/latest/debugging.html#fast-dev-run
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_string('dataset', 'rotten_tomatoes', '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_integer('seq_length', 20, '')
flags.DEFINE_float('lr', 1e-4, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
# you might need to change this depending on your machine
flags.DEFINE_integer('num_workers', 8, '')

FLAGS = flags.FLAGS

# clears the logs for you
sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')


# define the module
# all functions and parts of the code to be implemented are marked with *********Implement**************
class RTSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # *********Implement**************
        # initialize your model here and make use of the pre-trained BERT model defined in FLAGS.model
        # further define your loss function here. leverage the pytorch library for this purpose

        #load pre trained BERT model
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)

        #define Cross-entropy loss
        self.loss = th.nn.CrossEntropyLoss(reduction='none') 
        
    # this function prepares the data for you and uses the tokenizer from the pretrained model
    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model) #tokenizer

        def _tokenize(x):
            x['input_ids'] = tokenizer.batch_encode_plus(
                    x['text'], 
                    max_length=FLAGS.seq_length, 
                    pad_to_max_length=True)['input_ids']
            return x

        def _prepare_ds(split):
            #loading dataset from the nlp library
            ds = nlp.load_dataset(FLAGS.dataset, split=f'{split}')
            ds = ds.map(_tokenize, batched=True)
            ds.set_format(type='torch', columns=['input_ids', 'label']) #output sample type
            return ds

        self.train_ds, self.validation_ds, self.test_ds = map(_prepare_ds, ('train', 'validation', 'test'))

    # *********Implement**************
    # this function implements the forward step in your network
    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)
        return logits
    
    # *********Implement**************
    # this function defines the training step
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean() # batch loss for back propagation
        return {'loss': loss, 'log': {'train_loss': loss}}
        
    # *********Implement**************
    # this function defines the validation step
    # compute loss and accuracy for every batch
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': acc}
    
    # *********Implement**************
    # this function concludes the validation loop
    # aggregate loss and accuracy at every epoch end
    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}
        
    # *********Implement**************
    # this function defines the test step
    def test_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logits.argmax(-1) == batch['label']).float()
        return {'loss': loss, 'acc': acc}
        
    # *********Implement**************
    # this function concludes the test loop
    def test_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'test_loss': loss, 'test_acc': acc}
        return {**out, 'log': out}
        
        
    # this function defines the training data for you
    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=FLAGS.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=FLAGS.num_workers,
                )

    # this function defines the validation data for you
    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.validation_ds,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=FLAGS.num_workers,
                )
    
    # this function defines the test data for you
    def test_dataloader(self):
        return th.utils.data.DataLoader(
                self.test_ds,
                batch_size=FLAGS.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=FLAGS.num_workers,
                )
    
    
    # *********Implement**************
    # here you define the appropriate optimizer (use SGD the only one tested for this)
    # use the pytorch library for this
    # make sure to use the parameters defined in FLAGS
    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
        )

def main(_):
    # *********Implement**************
    # initialize your model and trainer here
    # further fit the model and don't forget to run the test; pytorch lightning does not automatically do that for you!
    # tensorboard logger to monitor training loss, validation loss and accuracy. This helps to determine whether model is overfitting or underfittting data.
    model = RTSentimentClassifier()
    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='rotten_tomatoes', version=0), #tensorboard logger 
    )
    trainer.fit(model) # run full training
    trainer.save_checkpoint("sample.ckpt")
    trainer.test(model) # run the test set seperately

if __name__ == '__main__':
    app.run(main)
