const Parse = require('parse/node');
const btoa = require('btoa');
const fs = require('fs');

Parse.serverURL = 'https://optoswim.back4app.io'; // This is your Server URL
Parse.initialize(
  '03Pq0kbLRvci8D3OV92OFIIbNidw3kZrGma2sruS', // This is your Application ID
  'SMnIF1sMs1zdczYwWU1SikdLtfIu4IzcWYhBhEMf', // This is your Javascript key
  'HSNUMKsUeTtNjTxOIPB2ct3FIiD6NMJp7yc5w9WW' // This is your Master key (never use it in the frontend)
);

const Images = Parse.Object.extend('Images');
const image = new Images();

image.set('swimDetected', true);
image.set('numberSwimmers', 1);
image.set('drownDetected', true);
image.set(
  'image',
  new Parse.File('resume.txt', { base64: btoa('resume.txt') })
);

image.save().then(
  (result) => {
    if (typeof document !== 'undefined')
      document.write(`Images created: ${JSON.stringify(result)}`);
    console.log('Images created', result);
  },
  (error) => {
    if (typeof document !== 'undefined')
      document.write(`Error while creating Images: ${JSON.stringify(error)}`);
    console.error('Error while creating Images: ', error);
  }
);

