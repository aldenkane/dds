const Parse = require('parse/node')
const btoa = require('btoa')
const fs = require('fs')

//Initializes Parse Object
Parse.serverURL = 'https://optoswimeye.back4app.io' // Server URL
Parse.initialize(
	'fqPHmhq9BPryvJRMRSyMRx974hOrK1KKdyKlUokV', // Application ID
	'9WV8pcSdsMg529URvKzTatId7iq4lwFRIItopkQI', // Javascript key
	'1ADOjf5x6kwRx1iXpI1ON5vC2lIIp60Yv9Dn5dSu' // Master key (never use it in the frontend)
)

// Creates an Image object in Parse DB
function sendImage(swimDetected, numberSwimmers, drownDetected, serialNo) {
	let file = fs.readFileSync('../last_Image/last_Frame.jpg')
	const Images = Parse.Object.extend('Images')
	const image = new Images()

	image.set('swimDetected', swimDetected)
	image.set('numberSwimmers', numberSwimmers)
	image.set('drownDetected', drownDetected)
	image.set('serialNo', serialNo)
	image.set('image', new Parse.File('last_Frame.jpg', { base64: btoa(file) }))

	image.save().then(
		(result) => {
			if (typeof document !== 'undefined')
				document.write(`Images created: ${JSON.stringify(result)}`)
			console.log('Images created', result)
		},
		(error) => {
			if (typeof document !== 'undefined')
				document.write(`Error while creating Images: ${JSON.stringify(error)}`)
			console.error('Error while creating Images: ', error)
		}
	)
}

//Creates an Events Object in Parse DB
function sendEvent(swimDetected, numberSwimmers, drownDetected, serialNo) {
	let file = fs.readFileSync('../last_Image/last_Frame.jpg')
	const Events = Parse.Object.extend('Events')
	const event = new Events()

	event.set('swimDetected', swimDetected)
	event.set('numberSwimmers', numberSwimmers)
	event.set('drownDetected', drownDetected)
	event.set('serialNo', serialNo)
	event.set('image', new Parse.File('last_Frame.jpg', { base64: btoa(file) }))

	event.save().then(
		(result) => {
			if (typeof document !== 'undefined')
				document.write(`Images created: ${JSON.stringify(result)}`)
			console.log('Images created', result)
		},
		(error) => {
			if (typeof document !== 'undefined')
				document.write(`Error while creating Images: ${JSON.stringify(error)}`)
			console.error('Error while creating Images: ', error)
		}
	)
}

// constantly watches and creates Events and Image object when detected
var prevEvent
const event = async () => {
	fs.watch('../last_Image/event.json', (event, filename) => {
		if (filename) {
			let jsObj = require('../last_Image/event.json')
			if (jsObj.swimDetected === true && prevEvent !== jsObj) {
				console.log('event detected')
				sendEvent(
					jsObj.swimDetected,
					parseInt(jsObj.numberSwimmers),
					jsObj.drownDetected,
					jsObj.serialNo
				)
				prevEvent = jsObj
			}
		}
	})
}

event()

// Starts watching and creating Images objects when a LiveFeed object is created
const liveQuery = async () => {
	const setLive
	var client = await new Parse.LiveQueryClient({
		applicationId: 'fqPHmhq9BPryvJRMRSyMRx974hOrK1KKdyKlUokV',
		serverURL: 'wss://' + 'optoswimeye.back4app.io', // Example: 'wss://livequerytutorial.back4app.io'
		javascriptKey: '9WV8pcSdsMg529URvKzTatId7iq4lwFRIItopkQI',
		masterKey: '1ADOjf5x6kwRx1iXpI1ON5vC2lIIp60Yv9Dn5dSu',
	})
	await client.open()

	// Creates a new Query object to help us fetch MyCustomClass objects
	const query = new Parse.Query('LiveFeed')
	query.equalTo('liveFeed', true)

	const query1 = new Parse.Query('LiveFeed')
	query.equalTo('liveFeed', false)

	var subscription = await client.subscribe(query)
	var subscription1 = await client.subscribe(query1)

	subscription.on('create', async () => {
		setLive = true
		fs.watch('../last_Image/event.json', (event, filename) => {
			if (setLive === true) {
				if (filename) {
					let jsObj = require('../last_Image/event.json')
					sendImage(
						jsObj.swimDetected,
						parseInt(jsObj.numberSwimmers),
						jsObj.drownDetected,
						jsObj.serialNo
					)
				}
			}
		})
	})

	subscription.on('update', async () => {
		// console.log('watcher started')
		setLive = false
	})
}

liveQuery()
