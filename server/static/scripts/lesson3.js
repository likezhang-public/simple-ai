'use strict';

class Lesson3 extends React.Component {
  constructor(props) {
    console.log(props)
    super(props);
    this.state = {
        value:"",
        result:""
    };

    this.handleInputChange = this.handleInputChange.bind(this);
    this.getPredict = this.getPredict.bind(this);
    this.updateTalkList = this.updateTalkList.bind(this);
  }

  handleInputChange(event) {
    event.preventDefault();   
    let v = event.target.value;  
    this.setState({value: v});
  }

  updateTalkList(sender, newLine) {
    let newTalks = this.state.result
    newTalks += sender + ": " + newLine + "\n"
    this.setState({result: newTalks})
  }

  getPredict() {
    let data = {
        data: {
            user_input: this.state.value
        }
    }

    this.updateTalkList("you", this.state.value)

    axios
    .post('/lesson3/chatbot', data)
    .then(response => {
        var data = response.data.result;
        console.log(data)
        this.updateTalkList("机器人", data)
    })
    .catch(function (error) { // 请求失败处理
        console.log(error);
    });

  }

  render() {
    return (
        <div>
            Lesson 3: 智障AI机器人
            <hr/>
            <input type="text" value={this.state.value} onChange={this.handleInputChange}/>
            <br/>
            <button onClick={() => this.getPredict() }>
              发送
            </button>
            <br/>
            <pre>
            { this.state.result }
            </pre>
        </div>
      );      
 }
}
