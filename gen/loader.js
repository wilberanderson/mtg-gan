var folders = ["Creature_B","Creature_G","Creature_R","Creature_U","Creature_W",
"Enchantment_B","Enchantment_G","Enchantment_R","Enchantment_U","Enchantment_W",
"Instant_B","Instant_G","Instant_R","Instant_U","Instant_W",
"Land_B","Land_G","Land_R","Land_U","Land_W",
"Sorcery_B","Sorcery_G","Sorcery_R","Sorcery_U","Sorcery_W"]
index = 0
hist = ['Creature_B/0_02_01.png']
function getRandomInt(max) {
  return Math.floor(Math.random() * max);
}
function getRadioValue() {
    out = ["",""]
    var ele = document.getElementsByTagName('input');
      
    for(i = 0; i < ele.length; i++) {
        if(ele[i].name==="color") {
          if(ele[i].checked)
            out[1] = ele[i].value
        }
        if(ele[i].name==="type") {
          if(ele[i].checked)
            out[0] = ele[i].value
        }
    }
    return out
}
function loadrandom() {
  tps = getRadioValue()
  folder = tps[0] + '_' + tps[1]
  img = folder + '/' +
        ((getRandomInt(19))) +'_' +
        ("0"+(getRandomInt(24)+1)).slice(-2) + '_' +
        ("0"+((folders.indexOf(folder)+1) )).slice(-2) + ".png"
  document.getElementById('image').src = (img)
  hist.push(img)
  index = hist.length - 1
}
function bluron() {
  document.getElementById('image').style.filter = 'blur(1.5px)';
}
function bluroff() {
  document.getElementById('image').style.filter = 'blur(0)';
}
function last(){
  if (index > 1){
    index--
    document.getElementById('image').src = hist[index]
  }
}
function next(){
  if (index < hist.length-1){
    index++
    document.getElementById('image').src = hist[index]
  }  
}